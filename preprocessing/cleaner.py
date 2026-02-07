import re
from typing import List, Dict


def clean_pages(pages: List[Dict]) -> List[Dict]:
    """
    Clean extracted PDF pages while preserving scientific content.

    Parameters
    ----------
    pages : List[Dict]
        Page dictionaries with keys: 'page_num', 'text'

    Returns
    -------
    List[Dict]
        Cleaned pages with same structure.
    """

    cleaned = []

    for page in pages:
        text = page["text"]

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common boilerplate patterns
        text = re.sub(r"arXiv:\d+\.\d+(v\d+)?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"©.*?All rights reserved\.", "", text, flags=re.IGNORECASE)

        cleaned.append(
            {
                "page_num": page["page_num"],
                "text": text.strip(),
            }
        )

    return cleaned
