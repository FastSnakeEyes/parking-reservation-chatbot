import argparse
from pathlib import Path

from app.ai.guardrails import get_pii_guard
from app.constants import STATIC_DOCS_DIR
from app.dependencies import create_vector_repository
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class MarkdownLoader:
    def __init__(self, directory: Path):
        self.directory = directory

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "title"),
                ("##", "section"),
                ("###", "subsection"),
            ],
            strip_headers=False,
        )

        self.size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

    def load(self) -> list[Document]:
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)

        documents = []

        for path in sorted(self.directory.rglob("*.md")):
            sections = self.header_splitter.split_text(path.read_text(encoding="utf-8"))

            chunks = self.size_splitter.split_documents(sections)
            source = path.relative_to(self.directory).as_posix()

            for index, document in enumerate(chunks):
                document.metadata.update(
                    {
                        "source": source,
                        "chunk_index": index,
                        "id": f"{source}:{index}",
                    }
                )

            documents.extend(chunks)

        return documents


def redact_documents(documents: list[Document]) -> None:
    """Mask PII in place before it ever reaches Milvus - defense in depth
    beyond the query-time and answer-time redaction in app/ai/guardrails.py,
    so even a raw store dump stays clean. settings.public_contacts is
    still honored, so the operator's own published phone/email survives."""
    guard = get_pii_guard()
    redacted_count = 0

    for document in documents:
        redacted = guard.redact(document.page_content)

        if redacted != document.page_content:
            redacted_count += 1
            document.page_content = redacted

    if redacted_count:
        print(f"Redacted PII in {redacted_count} of {len(documents)} chunks")


def main():
    parser = argparse.ArgumentParser(description="Load public static Markdown into Milvus")

    parser.add_argument(
        "--directory",
        type=Path,
        default=STATIC_DOCS_DIR,
    )

    args = parser.parse_args()
    documents = MarkdownLoader(args.directory).load()

    if not documents:
        raise ValueError("No Markdown content found")

    redact_documents(documents)

    repository = create_vector_repository()

    try:
        # Embeds documents, creates the collection, and upserts chunks.
        ids = repository.add_documents(documents)
        print(f"Stored {len(ids)} chunks")
    finally:
        repository.close()


if __name__ == "__main__":
    main()
