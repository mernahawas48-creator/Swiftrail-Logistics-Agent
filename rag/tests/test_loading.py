from rag.loading.loader import CorpusLoader


def test_loader_reads_all_seven_corpus_documents():
    documents = CorpusLoader().load()

    assert len(documents) == 7

    document_ids = {
        document.metadata["doc_id"]
        for document in documents
    }

    assert document_ids == {
        "credit_hold_policy",
        "rate_exception_policy",
        "portfolio_risk_guidelines",
        "invoice_collection_sop",
        "employee_access_policy",
        "shipment_pricing_reference",
        "delivery_exception_policy",
    }

    for document in documents:
        assert document.text
        assert "\r" not in document.text
        assert len(document.checksum) == 64
        assert document.source_file.exists()
