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


def test_loader_moves_front_matter_into_metadata(tmp_path):
    (tmp_path / "old_town.md").write_text(
        "---\ndoc_type: parking_details\nparking_id: old-town\n---\n"
        "# Old Town\n\n## Facilities\n- Height limit: 2.10 m\n"
    )

    documents = MarkdownLoader(tmp_path).load()

    assert len(documents) == 1  # no chunk of its own for the front matter
    assert "doc_type" not in documents[0].page_content
    assert documents[0].metadata["parking_id"] == "old-town"


def test_every_chunk_carries_its_heading_path(tmp_path):
    (tmp_path / "old_town.md").write_text(
        "# Old Town\n\n## Overview\nNear the Cathedral.\n\n## Facilities\n- Height limit: 2.10 m\n"
    )

    documents = MarkdownLoader(tmp_path).load()

    facilities = documents[1].page_content
    assert facilities.startswith("# Old Town\n## Facilities\n")  # the section still says which car park
    assert "Height limit: 2.10 m" in facilities


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
