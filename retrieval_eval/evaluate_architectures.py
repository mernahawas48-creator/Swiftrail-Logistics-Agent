"""Run the same fixed questions through Naive, Hybrid, and Agentic RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag.agentic_rag.controller import AgenticRAG
from rag.hybrid_rag.pipeline import HybridRAG
from rag.naive_rag.generator import GenerationUsage
from rag.naive_rag.pipeline import NaiveRAG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = (
    Path(__file__).resolve().parent / "questions.json"
)
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results"
)
SAFE_PREFIX = "i could not find enough authorized"


class ProviderQuotaExhausted(RuntimeError):
    """Raised when the model provider reports a non-retryable quota limit."""



@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    category: str
    query: str
    role: str
    top_k: int
    expected_section_ids: tuple[str, ...]
    required_term_groups: tuple[tuple[str, ...], ...]
    expected_abstain: bool
    forbidden_section_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaseResult:
    architecture: str
    case_id: str
    category: str
    correct: bool
    query: str
    role: str
    answer: str
    source_section_ids: tuple[str, ...]
    verification_passed: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    retrieval_attempts: int
    transient_api_retries: int
    retry_wait_seconds: float
    reason: str


@dataclass(frozen=True, slots=True)
class ArchitectureSummary:
    architecture: str
    correct: int
    total: int
    accuracy: float
    avg_input_tokens_per_query: float
    avg_output_tokens_per_query: float
    avg_total_tokens_per_query: float
    avg_latency_seconds_per_query: float
    safe_abstentions: int
    avg_retrieval_attempts: float
    total_transient_api_retries: int


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    model_name: str
    case_count: int
    summaries: tuple[ArchitectureSummary, ...]
    cases: tuple[CaseResult, ...]


def load_cases(
    path: str | Path = DEFAULT_QUESTIONS,
) -> tuple[EvaluationCase, ...]:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(payload, list):
        raise TypeError(
            "The architecture evaluation dataset must be a JSON list."
        )

    cases = tuple(
        EvaluationCase(
            case_id=str(item["case_id"]),
            category=str(item["category"]),
            query=str(item["query"]).strip(),
            role=str(item["role"]),
            top_k=int(item.get("top_k", 3)),
            expected_section_ids=tuple(
                str(value)
                for value in item.get(
                    "expected_section_ids",
                    [],
                )
            ),
            required_term_groups=tuple(
                tuple(str(value) for value in group)
                for group in item.get(
                    "required_term_groups",
                    [],
                )
            ),
            expected_abstain=bool(
                item.get("expected_abstain", False)
            ),
            forbidden_section_ids=tuple(
                str(value)
                for value in item.get(
                    "forbidden_section_ids",
                    [],
                )
            ),
        )
        for item in payload
    )

    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Architecture evaluation case IDs must be unique."
        )

    return cases


def build_architectures() -> tuple[tuple[str, Any], ...]:
    """Construct the three required complete answer pipelines."""

    return (
        ("Naive RAG", NaiveRAG()),
        ("Hybrid RAG", HybridRAG()),
        ("Agentic RAG", AgenticRAG(max_attempts=2)),
    )


def run_comparison(
    *,
    cases: Sequence[EvaluationCase],
    architectures: Sequence[tuple[str, Any]],
    max_api_retries: int = 5,
    initial_retry_delay: float = 5.0,
    inter_case_delay: float = 1.5,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
) -> ComparisonReport:
    """Run every architecture against every fixed test question.

    Transient provider 429/5xx errors are retried without changing the fixed
    questions or the architecture configuration. Retry waiting time is
    recorded separately and is not included in the per-query latency metric.

    Completed architecture/case pairs can be checkpointed after every
    successful case. If the provider quota is exhausted, a later run can
    resume from the checkpoint without repeating completed model calls.
    """

    if max_api_retries < 0:
        raise ValueError("max_api_retries cannot be negative.")

    if initial_retry_delay < 0:
        raise ValueError("initial_retry_delay cannot be negative.")

    if inter_case_delay < 0:
        raise ValueError("inter_case_delay cannot be negative.")

    resolved_checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else None
    )
    dataset_signature = _dataset_signature(cases)

    all_results: list[CaseResult] = []
    model_name = "unknown"

    if (
        resume
        and resolved_checkpoint is not None
        and resolved_checkpoint.exists()
    ):
        (
            checkpoint_signature,
            checkpoint_model,
            checkpoint_results,
        ) = _load_checkpoint(
            resolved_checkpoint
        )

        if checkpoint_signature != dataset_signature:
            raise ValueError(
                "The saved evaluation checkpoint was created from a "
                "different questions dataset. Do not change questions.json "
                "between architecture runs."
            )

        all_results.extend(checkpoint_results)
        model_name = checkpoint_model

        print(
            f"Resuming from checkpoint: "
            f"{len(all_results)} completed architecture/case runs.",
            flush=True,
        )

    completed_keys = {
        (item.architecture, item.case_id)
        for item in all_results
    }

    for architecture_name, pipeline in architectures:
        _warm_embedding(pipeline)

        generator = getattr(
            pipeline,
            "generator",
            None,
        )
        current_model_name = str(
            getattr(
                generator,
                "model_name",
                model_name,
            )
        )

        if (
            model_name != "unknown"
            and current_model_name != "unknown"
            and model_name != current_model_name
        ):
            raise ValueError(
                "The saved checkpoint uses model "
                f"'{model_name}', but the current run uses "
                f"'{current_model_name}'. Keep the same model for a fair "
                "architecture comparison."
            )

        model_name = current_model_name

        for case_index, case in enumerate(cases, start=1):
            key = (architecture_name, case.case_id)

            if key in completed_keys:
                print(
                    f"[{architecture_name}] "
                    f"{case_index}/{len(cases)} "
                    f"{case.case_id} -- already completed, skipping",
                    flush=True,
                )
                continue

            print(
                f"[{architecture_name}] "
                f"{case_index}/{len(cases)} "
                f"{case.case_id}",
                flush=True,
            )

            (
                response,
                latency,
                transient_api_retries,
                retry_wait_seconds,
            ) = _answer_with_transient_retry(
                pipeline=pipeline,
                generator=generator,
                case=case,
                max_api_retries=max_api_retries,
                initial_retry_delay=initial_retry_delay,
            )

            usage = _usage(generator)
            source_sections = tuple(
                str(source.section_id)
                for source in response.sources
            )
            verification_passed = (
                _verification_passed(response)
            )
            correct, reason = score_answer(
                case,
                response.answer,
                source_sections,
                verification_passed,
            )

            result = CaseResult(
                architecture=architecture_name,
                case_id=case.case_id,
                category=case.category,
                correct=correct,
                query=case.query,
                role=case.role,
                answer=response.answer,
                source_section_ids=source_sections,
                verification_passed=(
                    verification_passed
                ),
                input_tokens=(
                    usage.input_tokens
                ),
                output_tokens=(
                    usage.output_tokens
                ),
                total_tokens=(
                    usage.total_tokens
                ),
                latency_seconds=latency,
                retrieval_attempts=int(
                    getattr(
                        response,
                        "attempts",
                        1,
                    )
                ),
                transient_api_retries=(
                    transient_api_retries
                ),
                retry_wait_seconds=(
                    retry_wait_seconds
                ),
                reason=reason,
            )
            all_results.append(result)
            completed_keys.add(key)

            if resolved_checkpoint is not None:
                _save_checkpoint(
                    resolved_checkpoint,
                    dataset_signature=dataset_signature,
                    model_name=model_name,
                    results=all_results,
                )

            if (
                inter_case_delay > 0
                and case_index < len(cases)
            ):
                time.sleep(inter_case_delay)

    summaries = tuple(
        _summarize(
            architecture_name,
            [
                item
                for item in all_results
                if item.architecture
                == architecture_name
            ],
        )
        for architecture_name, _ in architectures
    )

    report = ComparisonReport(
        model_name=model_name,
        case_count=len(cases),
        summaries=summaries,
        cases=tuple(all_results),
    )

    if (
        resolved_checkpoint is not None
        and resolved_checkpoint.exists()
    ):
        resolved_checkpoint.unlink()

    return report



def _answer_with_transient_retry(
    *,
    pipeline: Any,
    generator: Any,
    case: EvaluationCase,
    max_api_retries: int,
    initial_retry_delay: float,
) -> tuple[Any, float, int, float]:
    """Execute one case and retry only transient provider/API failures.

    The returned latency measures the successful pipeline attempt. Deliberate
    retry waiting is tracked separately so service congestion does not inflate
    the architecture latency comparison.
    """

    retry_count = 0
    total_wait = 0.0

    while True:
        _reset_usage(generator)
        started = time.perf_counter()

        try:
            response = pipeline.answer(
                case.query,
                role=case.role,
                top_k=case.top_k,
            )
            latency = time.perf_counter() - started

            return (
                response,
                latency,
                retry_count,
                total_wait,
            )
        except Exception as exc:
            if _is_non_retryable_quota_exhausted(exc):
                raise ProviderQuotaExhausted(
                    "The model-provider quota is exhausted for the "
                    f"current model while running case '{case.case_id}'."
                ) from exc

            if not (
                _is_transient_service_error(exc)
                or _is_transient_rate_limit(exc)
            ):
                raise

            if retry_count >= max_api_retries:
                raise RuntimeError(
                    "The model provider remained temporarily unavailable after "
                    f"{max_api_retries} retries for case "
                    f"'{case.case_id}'. The fixed evaluation dataset was "
                    "not changed."
                ) from exc

            delay = initial_retry_delay * (2 ** retry_count)
            retry_count += 1
            total_wait += delay

            print(
                "  The model provider returned a transient API error. "
                f"Retry {retry_count}/{max_api_retries} "
                f"in {delay:.1f}s...",
                flush=True,
            )

            if delay > 0:
                time.sleep(delay)


def _status_code(exc: Exception) -> int | None:
    """Extract an HTTP-like status code from common provider exceptions."""

    status_code = getattr(
        exc,
        "status_code",
        None,
    )

    if isinstance(status_code, int):
        return status_code

    code = getattr(
        exc,
        "code",
        None,
    )

    if isinstance(code, int):
        return code

    message = str(exc).upper()

    for candidate in (429, 500, 502, 503, 504):
        if str(candidate) in message:
            return candidate
    return None



def _is_transient_service_error(exc: Exception) -> bool:
    """Recognize retryable provider/server failures."""

    return _status_code(exc) in {500, 502, 503, 504}


def _is_non_retryable_quota_exhausted(exc: Exception) -> bool:
    """Distinguish exhausted account quota from a temporary rate limit."""

    message = str(exc).lower()

    return (
        _status_code(exc) == 429
        and "quota" in message
        and any(term in message for term in ("exhausted", "exceeded", "insufficient"))
    )


def _is_transient_rate_limit(exc: Exception) -> bool:
    """Treat a non-quota 429 response as a retryable rate limit."""

    return (
        _status_code(exc) == 429
        and not _is_non_retryable_quota_exhausted(exc)
    )


def _dataset_signature(
    cases: Sequence[EvaluationCase],
) -> str:
    """Hash the fixed case definitions so a resumed run cannot change them."""

    payload = [
        asdict(case)
        for case in cases
    ]
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def _save_checkpoint(
    path: Path,
    *,
    dataset_signature: str,
    model_name: str,
    results: Sequence[CaseResult],
) -> None:
    """Persist completed case results atomically after every successful case."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "dataset_signature": dataset_signature,
        "model_name": model_name,
        "results": [
            asdict(item)
            for item in results
        ],
    }

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
) -> tuple[
    str,
    str,
    list[CaseResult],
]:
    """Load completed results from a previous quota-interrupted run."""

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    results = [
        CaseResult(
            architecture=str(item["architecture"]),
            case_id=str(item["case_id"]),
            category=str(item["category"]),
            correct=bool(item["correct"]),
            query=str(item["query"]),
            role=str(item["role"]),
            answer=str(item["answer"]),
            source_section_ids=tuple(
                str(value)
                for value in item["source_section_ids"]
            ),
            verification_passed=bool(
                item["verification_passed"]
            ),
            input_tokens=int(item["input_tokens"]),
            output_tokens=int(item["output_tokens"]),
            total_tokens=int(item["total_tokens"]),
            latency_seconds=float(
                item["latency_seconds"]
            ),
            retrieval_attempts=int(
                item["retrieval_attempts"]
            ),
            transient_api_retries=int(
                item.get(
                    "transient_api_retries",
                    0,
                )
            ),
            retry_wait_seconds=float(
                item.get(
                    "retry_wait_seconds",
                    0.0,
                )
            ),
            reason=str(item["reason"]),
        )
        for item in payload.get(
            "results",
            []
        )
    ]

    return (
        str(payload["dataset_signature"]),
        str(payload.get("model_name", "unknown")),
        results,
    )


def score_answer(
    case: EvaluationCase,
    answer: str,
    source_sections: Sequence[str],
    verification_passed: bool,
) -> tuple[bool, str]:
    """Apply the fixed answer rubric for one test case."""

    normalized_answer = _normalize(answer)
    source_set = set(source_sections)

    leaked_sections = (
        set(case.forbidden_section_ids)
        .intersection(source_set)
    )
    if leaked_sections:
        return (
            False,
            "Forbidden source section retrieved: "
            + ", ".join(sorted(leaked_sections)),
        )

    abstained = normalized_answer.startswith(
        SAFE_PREFIX
    )

    if case.expected_abstain:
        if abstained:
            return True, "Safe abstention expected and returned."
        return False, "The case required a safe abstention."

    if abstained:
        return False, "The pipeline abstained on an answerable case."

    if not verification_passed:
        return False, "The final answer failed verification."

    missing_sections = (
        set(case.expected_section_ids)
        .difference(source_set)
    )
    if missing_sections:
        return (
            False,
            "Missing expected source section(s): "
            + ", ".join(sorted(missing_sections)),
        )

    for group in case.required_term_groups:
        if not any(
            _normalize(option)
            in normalized_answer
            for option in group
        ):
            return (
                False,
                "Missing required answer evidence: "
                + " OR ".join(group),
            )

    return True, "Passed the fixed answer rubric."


def write_report(
    report: ComparisonReport,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        directory
        / "architecture_comparison.json"
    )
    markdown_path = (
        directory
        / "architecture_comparison.md"
    )

    json_path.write_text(
        json.dumps(
            asdict(report),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown_path.write_text(
        _markdown(report),
        encoding="utf-8",
    )

    return json_path, markdown_path


def _summarize(
    architecture_name: str,
    results: Sequence[CaseResult],
) -> ArchitectureSummary:
    total = len(results)
    correct = sum(
        1 for item in results
        if item.correct
    )

    return ArchitectureSummary(
        architecture=architecture_name,
        correct=correct,
        total=total,
        accuracy=(
            correct / total
            if total
            else 0.0
        ),
        avg_input_tokens_per_query=_mean(
            [
                item.input_tokens
                for item in results
            ]
        ),
        avg_output_tokens_per_query=_mean(
            [
                item.output_tokens
                for item in results
            ]
        ),
        avg_total_tokens_per_query=_mean(
            [
                item.total_tokens
                for item in results
            ]
        ),
        avg_latency_seconds_per_query=_mean(
            [
                item.latency_seconds
                for item in results
            ]
        ),
        safe_abstentions=sum(
            1
            for item in results
            if _normalize(item.answer).startswith(
                SAFE_PREFIX
            )
        ),
        avg_retrieval_attempts=_mean(
            [
                item.retrieval_attempts
                for item in results
            ]
        ),
        total_transient_api_retries=sum(
            item.transient_api_retries
            for item in results
        ),
    )


def _verification_passed(
    response: Any,
) -> bool:
    if hasattr(
        response,
        "verification_passed",
    ):
        return bool(
            response.verification_passed
        )

    verification = getattr(
        response,
        "verification",
        None,
    )

    if verification is None:
        return False

    return bool(
        getattr(
            verification,
            "passed",
            False,
        )
    )


def _reset_usage(generator: Any) -> None:
    reset = getattr(
        generator,
        "reset_usage",
        None,
    )
    if callable(reset):
        reset()


def _usage(generator: Any) -> GenerationUsage:
    usage = getattr(
        generator,
        "usage_totals",
        None,
    )

    if isinstance(usage, GenerationUsage):
        return usage

    return GenerationUsage()


def _warm_embedding(pipeline: Any) -> None:
    """Load the embedding model before latency measurement without calling Mistral."""

    candidates = [
        getattr(pipeline, "embedder", None),
        getattr(
            getattr(pipeline, "searcher", None),
            "embedder",
            None,
        ),
        getattr(
            getattr(pipeline, "retriever", None),
            "embedder",
            None,
        ),
    ]

    for embedder in candidates:
        if embedder is not None:
            embedder.embed_query(
                "Swiftrail retrieval evaluation warmup"
            )
            return


def _normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower().replace("%", " percent "),
    ).strip()


def _mean(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _markdown(
    report: ComparisonReport,
) -> str:
    lines = [
        "# Swiftrail Retrieval Architecture Comparison",
        "",
        f"Model: `{report.model_name}`",
        f"Fixed test cases: {report.case_count}",
        "",
        "| Architecture | Correct / Total | Accuracy | Avg. input tokens/query | Avg. output tokens/query | Avg. total tokens/query | Avg. latency/query | Avg. retrieval attempts | Safe abstentions | Transient API retries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in report.summaries:
        lines.append(
            "| "
            f"{summary.architecture} | "
            f"{summary.correct}/{summary.total} | "
            f"{summary.accuracy:.1%} | "
            f"{summary.avg_input_tokens_per_query:.1f} | "
            f"{summary.avg_output_tokens_per_query:.1f} | "
            f"{summary.avg_total_tokens_per_query:.1f} | "
            f"{summary.avg_latency_seconds_per_query:.3f}s | "
            f"{summary.avg_retrieval_attempts:.2f} | "
            f"{summary.safe_abstentions} | "
            f"{summary.total_transient_api_retries} |"
        )

    lines.extend(
        [
            "",
            "## Per-case results",
            "",
            "| Architecture | Case | Category | Correct | Sources | Verification | Attempts | API retries | Latency | Reason |",
            "|---|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )

    for item in report.cases:
        lines.append(
            "| "
            f"{item.architecture} | "
            f"{item.case_id} | "
            f"{item.category} | "
            f"{'yes' if item.correct else 'no'} | "
            f"{', '.join(item.source_section_ids) or '-'} | "
            f"{'pass' if item.verification_passed else 'fail'} | "
            f"{item.retrieval_attempts} | "
            f"{item.transient_api_retries} | "
            f"{item.latency_seconds:.3f}s | "
            f"{item.reason.replace('|', '/')} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the three required Swiftrail RAG architectures "
            "on the same fixed answer-level test set."
        )
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--max-api-retries",
        type=int,
        default=5,
        help=(
            "Retries for transient provider 429/5xx errors "
            "per evaluation case."
        ),
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help=(
            "Initial retry delay in seconds. Each retry doubles "
            "the previous delay."
        ),
    )
    parser.add_argument(
        "--inter-case-delay",
        type=float,
        default=1.5,
        help=(
            "Pause between fixed cases to reduce burst pressure. "
            "This pause is excluded from latency metrics."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint path. Defaults to a hidden JSON file inside "
            "the output directory."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore any saved checkpoint and restart the comparison."
        ),
    )
    args = parser.parse_args()

    cases = load_cases(args.questions)

    print(
        f"Running {len(cases)} fixed cases "
        "through Naive, Hybrid, and Agentic RAG..."
    )
    print(
        "This command calls Mistral for answerable cases. "
        "Do not edit questions.json between architecture runs."
    )

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else (
            args.output_dir
            / ".architecture_comparison_checkpoint.json"
        )
    )

    try:
        report = run_comparison(
            cases=cases,
            architectures=build_architectures(),
            max_api_retries=args.max_api_retries,
            initial_retry_delay=args.retry_delay,
            inter_case_delay=args.inter_case_delay,
            checkpoint_path=checkpoint_path,
            resume=not args.no_resume,
        )
    except ProviderQuotaExhausted as exc:
        print("\nEvaluation paused because the model-provider quota was reached.")
        print(str(exc))
        print(
            "Completed results were saved to: "
            f"{checkpoint_path}"
        )
        print(
            "After the quota resets, run the same command again. "
            "Completed architecture/case pairs will be skipped automatically."
        )
        return

    json_path, markdown_path = write_report(
        report,
        args.output_dir,
    )

    print("\nFinal comparison:")
    for summary in report.summaries:
        print(
            f"- {summary.architecture}: "
            f"{summary.correct}/{summary.total} "
            f"({summary.accuracy:.1%}), "
            f"avg tokens={summary.avg_total_tokens_per_query:.1f}, "
            f"avg latency={summary.avg_latency_seconds_per_query:.3f}s, "
            f"avg attempts={summary.avg_retrieval_attempts:.2f}, "
            f"API retries={summary.total_transient_api_retries}"
        )

    print(f"\nJSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
