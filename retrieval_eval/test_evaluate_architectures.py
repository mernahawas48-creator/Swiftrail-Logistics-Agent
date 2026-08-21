from __future__ import annotations

from types import SimpleNamespace

import pytest

from retrieval_eval.evaluate_architectures import (
    EvaluationCase,
    ProviderQuotaExhausted,
    _answer_with_transient_retry,
    _dataset_signature,
    _is_non_retryable_quota_exhausted,
    _is_transient_service_error,
    _load_checkpoint,
    _save_checkpoint,
    score_answer,
)


def _case(**overrides):
    values = {
        "case_id": "case",
        "category": "test",
        "query": "Who can release it?",
        "role": "finance_manager",
        "top_k": 3,
        "expected_section_ids": ("CH-3",),
        "required_term_groups": (
            ("finance manager",),
        ),
        "expected_abstain": False,
        "forbidden_section_ids": (),
    }
    values.update(overrides)
    return EvaluationCase(**values)


def test_score_answer_requires_expected_source_and_terms():
    correct, reason = score_answer(
        _case(),
        "Only a finance manager may release it [1].",
        ("CH-3",),
        True,
    )

    assert correct
    assert "Passed" in reason


def test_score_answer_rejects_missing_expected_source():
    correct, _ = score_answer(
        _case(),
        "Only a finance manager may release it [1].",
        ("CH-2",),
        True,
    )

    assert not correct


def test_expected_abstention_is_scored_as_correct():
    correct, _ = score_answer(
        _case(
            expected_section_ids=(),
            required_term_groups=(),
            expected_abstain=True,
        ),
        (
            "I could not find enough authorized information "
            "in the Swiftrail knowledge base to answer this question."
        ),
        (),
        True,
    )

    assert correct

class _Transient503(Exception):
    status_code = 503


class _RetryPipeline:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def answer(self, query, *, role, top_k):
        self.calls += 1

        if self.calls <= self.failures:
            raise _Transient503(
                "503 UNAVAILABLE"
            )

        return SimpleNamespace(
            answer="grounded answer",
            sources=(),
        )


class _UsageGenerator:
    def reset_usage(self):
        return None


def test_transient_service_error_is_detected():
    assert _is_transient_service_error(
        _Transient503("503 UNAVAILABLE")
    )
    assert not _is_transient_service_error(
        RuntimeError("400 BAD REQUEST")
    )


def test_case_retries_transient_503_and_then_succeeds(
    monkeypatch,
):
    monkeypatch.setattr(
        "retrieval_eval.evaluate_architectures.time.sleep",
        lambda _: None,
    )

    pipeline = _RetryPipeline(failures=2)

    response, latency, retries, wait = (
        _answer_with_transient_retry(
            pipeline=pipeline,
            generator=_UsageGenerator(),
            case=_case(),
            max_api_retries=3,
            initial_retry_delay=1.0,
        )
    )

    assert response.answer == "grounded answer"
    assert pipeline.calls == 3
    assert retries == 2
    assert wait == 3.0
    assert latency >= 0.0


def test_non_transient_error_is_not_hidden():
    class BrokenPipeline:
        def answer(self, query, *, role, top_k):
            raise ValueError("broken")

    with pytest.raises(
        ValueError,
        match="broken",
    ):
        _answer_with_transient_retry(
            pipeline=BrokenPipeline(),
            generator=_UsageGenerator(),
            case=_case(),
            max_api_retries=3,
            initial_retry_delay=0.0,
        )

class _Quota429(Exception):
    status_code = 429


def test_non_retryable_quota_is_detected_and_not_retried():
    exc = _Quota429("429 account quota exceeded")

    assert _is_non_retryable_quota_exhausted(exc)

    class QuotaPipeline:
        def __init__(self):
            self.calls = 0

        def answer(self, query, *, role, top_k):
            self.calls += 1
            raise exc

    pipeline = QuotaPipeline()

    with pytest.raises(ProviderQuotaExhausted):
        _answer_with_transient_retry(
            pipeline=pipeline,
            generator=_UsageGenerator(),
            case=_case(),
            max_api_retries=5,
            initial_retry_delay=0.0,
        )

    assert pipeline.calls == 1


def test_checkpoint_round_trip(tmp_path):
    case = _case()
    signature = _dataset_signature((case,))
    path = tmp_path / "checkpoint.json"

    from retrieval_eval.evaluate_architectures import CaseResult

    result = CaseResult(
        architecture="Naive RAG",
        case_id=case.case_id,
        category=case.category,
        correct=True,
        query=case.query,
        role=case.role,
        answer="grounded answer",
        source_section_ids=("CH-3",),
        verification_passed=True,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        latency_seconds=0.5,
        retrieval_attempts=1,
        transient_api_retries=0,
        retry_wait_seconds=0.0,
        reason="Passed.",
    )

    _save_checkpoint(
        path,
        dataset_signature=signature,
        model_name="mistral-test",
        results=[result],
    )

    loaded_signature, model_name, results = (
        _load_checkpoint(path)
    )

    assert loaded_signature == signature
    assert model_name == "mistral-test"
    assert results == [result]
