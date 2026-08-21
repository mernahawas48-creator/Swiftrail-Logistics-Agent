from rag.chunking.chunker import MarkdownChunker
from rag.loading.loader import CorpusLoader


def test_chunker_creates_section_aware_chunks():
    documents = CorpusLoader().load()

    chunks = MarkdownChunker(
        chunk_size=1000,
        overlap=120,
    ).chunk_documents(documents)

    assert len(documents) == 7
    assert len(chunks) == 27

    section_ids = {
        chunk.metadata.section_id
        for chunk in chunks
    }

    assert {
        "CH-1",
        "CH-3",
        "RE-1",
        "RE-2",
        "PR-2",
        "IC-3",
        "AC-2",
        "SP-4",
        "DR-3",
    }.issubset(section_ids)

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))

    for chunk in chunks:
        assert chunk.text
        assert chunk.metadata.doc_id
        assert chunk.metadata.section_id in chunk.text
        assert len(chunk.metadata.source_checksum) == 64


def test_chunk_ids_are_stable_between_runs():
    documents = CorpusLoader().load()
    chunker = MarkdownChunker(
        chunk_size=1000,
        overlap=120,
    )

    first_run = chunker.chunk_documents(documents)
    second_run = chunker.chunk_documents(documents)

    assert [
        chunk.chunk_id
        for chunk in first_run
    ] == [
        chunk.chunk_id
        for chunk in second_run
    ]
