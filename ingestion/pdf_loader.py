import fitz  # PyMuPDF
import os
import hashlib
from typing import Dict, List


class PDFLoader:
    """
    Responsible for loading a PDF and extracting clean, page-wise text.
    No chunking, no semantic processing, no hallucination control here.
    """

    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError("Input file must be a PDF")

        self.pdf_path = pdf_path
        self.doc_id = self._generate_doc_id(pdf_path)

    def _generate_doc_id(self, path: str) -> str:
        """
        Deterministic document ID based on file path.
        This guarantees reproducibility across runs.
        """
        return hashlib.md5(path.encode("utf-8")).hexdigest()

    def load(self) -> Dict:
        """
        Load PDF and extract page-wise text.
        Returns a structured dictionary.
        """
        document = fitz.open(self.pdf_path)

        pages: List[Dict] = []

        for page_index in range(len(document)):
            page = document[page_index]

            # Extract text in reading order
            text = page.get_text("text")

            # Light normalization ONLY
            text = self._normalize_text(text)

            pages.append({
                "page_num": page_index + 1,
                "text": text
            })

        document.close()

        return {
            "doc_id": self.doc_id,
            "source_path": self.pdf_path,
            "num_pages": len(pages),
            "pages": pages
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Minimal normalization.
        Heavy cleaning belongs to preprocessing stage.
        """
        # Replace multiple spaces with single space
        text = " ".join(text.split())

        # Normalize line breaks
        text = text.replace("\u00ad", "")  # soft hyphen

        return text.strip()


if __name__ == "__main__":
    # Manual sanity test
    sample_pdf = r"C:\ML\hallucination_controlled_rag\data\raw\pdfs\sample.pdf"

    loader = PDFLoader(sample_pdf)
    output = loader.load()

    print(f"Document ID: {output['doc_id']}")
    print(f"Number of pages: {output['num_pages']}")
    print("\n--- Page 1 Preview ---\n")
    print(output["pages"][0]["text"][:1000])
