import re
from typing import List

from healthcare_rag.data.loaders.base_loader import Document


class TextCleaner:

    def __init__(self):
        self.page_number_pattern = re.compile(
            r"\n\s*Page\s+\d+\s*\n",
            re.IGNORECASE,
        )

        self.header_pattern = re.compile(
            r"\n\s*Header\s*\n",
            re.IGNORECASE,
        )

        self.footer_pattern = re.compile(
            r"\n\s*Footer\s*\n",
            re.IGNORECASE,
        )

        self.url_pattern = re.compile(
            r"https?://\S+",
            re.IGNORECASE,
        )

        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )

    def clean(self, documents: List[Document]) -> List[Document]:
        cleaned_documents = []

        for document in documents:
            cleaned_content = self._clean_text(document.content)

            cleaned_documents.append(
                Document(
                    content=cleaned_content,
                    metadata=document.metadata.copy(),
                    source=document.source,
                )
            )

        return cleaned_documents

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = self.page_number_pattern.sub("\n", text)
        text = self.header_pattern.sub("\n", text)
        text = self.footer_pattern.sub("\n", text)

        text = self.url_pattern.sub("", text)
        text = self.email_pattern.sub("", text)

        # Normalize spaces without destroying newlines.
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize excessive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def remove_empty_chunks(
        self,
        documents: List[Document],
        min_length: int = 50,
    ) -> List[Document]:

        return [
            document
            for document in documents
            if document.content
            and len(document.content.strip()) >= min_length
        ]