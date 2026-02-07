import hashlib
from pathlib import Path
import fitz  # PyMuPDF


class PDFLoader:
    """
    Loads a user-uploaded PDF and extracts page-level text.
    """

    def __init__(self, pdf_path: Path):
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.pdf_path = pdf_path
        self.doc_id = self._compute_doc_id(pdf_path)

    def load(self):
        doc = fitz.open(self.pdf_path)
        pages = []

        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append(
                    {
                        "page_num": i + 1,
                        "text": text,
                    }
                )

        return {
            "doc_id": self.doc_id,
            "pages": pages,
        }

    def _compute_doc_id(self, path: Path) -> str:
        hasher = hashlib.sha256()
        hasher.update(path.read_bytes())
        return hasher.hexdigest()[:16]
