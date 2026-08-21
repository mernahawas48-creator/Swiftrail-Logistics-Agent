from rag.vector_store.config import VectorStoreSettings


def test_settings_read_environment_when_created(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_documents")
    monkeypatch.setenv("QDRANT_VECTOR_SIZE", "512")
    monkeypatch.setenv("QDRANT_TOP_K", "7")

    settings = VectorStoreSettings()

    assert settings.url == "http://qdrant.test:6333"
    assert settings.collection_name == "test_documents"
    assert settings.vector_size == 512
    assert settings.default_top_k == 7
