import argparse
import re
from pathlib import Path

from app.ai.guardrails import get_pii_guard
from app.constants import STATIC_DOCS_DIR
from app.dependencies import create_vector_repository
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

HEADERS = [("#", "title"), ("##", "section"), ("###", "subsection")]
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """`key: value` lines of a leading `---` block, and the text after it -
    otherwise the block would be indexed as a chunk of its own."""
    match = FRONT_MATTER.match(text)
    if not match:
        return {}, text

    fields = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return fields, text[match.end() :]


def with_heading_path(document: Document) -> Document:
    """Prefix a chunk with every heading above it, so a section such as
    "## Facilities" still says which car park it describes - for both the
    embedding and BM25 ("height limit Old Town")."""
    headings = [f"{marker} {document.metadata[key]}" for marker, key in HEADERS if key in document.metadata]
    document.page_content = "\n".join([*headings, document.page_content])
    return document


class MarkdownLoader:
    def __init__(self, directory: Path):
        self.directory = directory

        # Headers are stripped from the sections and added back to every chunk by with_heading_path.
        self.header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)

        self.size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

    def load(self) -> list[Document]:
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)

        documents = []

        for path in sorted(self.directory.rglob("*.md")):
            front_matter, text = split_front_matter(path.read_text(encoding="utf-8"))
            sections = self.header_splitter.split_text(text)

            chunks = [with_heading_path(chunk) for chunk in self.size_splitter.split_documents(sections)]
            source = path.relative_to(self.directory).as_posix()

            for index, document in enumerate(chunks):
                document.metadata.update(
                    {
                        **front_matter,
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
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and rebuild the collection (needed once to switch to hybrid search, "
        "and to remove chunks of docs that were shortened or deleted)",
    )

    args = parser.parse_args()
    documents = MarkdownLoader(args.directory).load()

    if not documents:
        raise ValueError("No Markdown content found")

    redact_documents(documents)

    repository = create_vector_repository()

    try:
        if args.recreate:
            repository.drop_collection()
            print("Dropped the collection; rebuilding it")

        # Embeds documents, creates the collection, and upserts chunks.
        ids = repository.add_documents(documents)
        print(f"Stored {len(ids)} chunks")
    finally:
        repository.close()


if __name__ == "__main__":
    main()
