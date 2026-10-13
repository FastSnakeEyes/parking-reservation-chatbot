"""Static-doc ingestion (scripts/ingest_static.py): chunking and the PII
redaction applied before anything reaches Milvus."""

import pytest
from app.config import settings
from app.constants import STATIC_DOCS_DIR
from langchain_core.documents import Document
from scripts.ingest_static import MarkdownLoader, redact_documents


def test_loader_chunks_markdown_with_source_metadata(tmp_path):
    (tmp_path / "rules.md").write_text("# Rules\n\n## Payment\nPay at the machine.\n\n## Exit\nUse the barrier.\n")

    documents = MarkdownLoader(tmp_path).load()

    assert [document.metadata["id"] for document in documents] == ["rules.md:0", "rules.md:1"]
    assert all(document.metadata["source"] == "rules.md" for document in documents)
    assert "Pay at the machine." in documents[0].page_content


def test_loader_reads_the_real_static_docs():
    documents = MarkdownLoader(STATIC_DOCS_DIR).load()

    sources = {document.metadata["source"] for document in documents}
    assert "general_info.md" in sources
    assert len(documents) > len(sources)  # long pages are split into several chunks


def test_loader_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        MarkdownLoader(tmp_path / "missing").load()


def test_redaction_masks_private_contacts_but_keeps_public_ones(monkeypatch):
    monkeypatch.setattr(settings, "public_contacts", ["info@vilpark.lt"])
    documents = [Document(page_content="Write to info@vilpark.lt or to jonas.petraitis@gmail.com.")]

    redact_documents(documents)

    assert "info@vilpark.lt" in documents[0].page_content
    assert "jonas.petraitis@gmail.com" not in documents[0].page_content
