# src/document_loader.py

import fitz                 # pymupdf
from pathlib import Path
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    doc  = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, source: str) -> tuple[list[str], list[str]]:
    """
    Split text into overlapping chunks.
    Returns (texts, sources) ready to pass into retriever.index()
    """
    texts   = []
    sources = []
    start   = 0

    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if len(chunk) > 50:          # skip tiny chunks
            texts.append(chunk)
            sources.append(source)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return texts, sources


def load_documents(data_dir: str = "data/") -> tuple[list[str], list[str]]:
    """Load all PDFs from a directory and return chunks."""
    all_texts   = []
    all_sources = []
    data_path   = Path(data_dir)

    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {data_dir}")
        return [], []

    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name}...")
        text = load_pdf(str(pdf_path))

        if not text.strip():
            print(f"  Warning: no text extracted from {pdf_path.name}")
            continue

        texts, sources = chunk_text(text, source=pdf_path.name)
        all_texts.extend(texts)
        all_sources.extend(sources)
        print(f"  → {len(texts)} chunks")

    print(f"\nTotal: {len(all_texts)} chunks from {len(pdf_files)} PDFs")
    return all_texts, all_sources

# if __name__ == "__main__":
#     texts, sources = load_documents("data/") 

#     print(f"\nFirst chunk:")
#     print(f"  Source: {sources[0]}")
#     print(f"  Text:   {texts[0][:200]}...")

#     print(f"\nLast chunk:")
#     print(f"  Source: {sources[-1]}")
#     print(f"  Text:   {texts[-1][:200]}...")