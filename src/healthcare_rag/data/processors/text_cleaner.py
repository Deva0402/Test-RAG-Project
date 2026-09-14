from importlib import metadata
import re
from typing import List
from xml.dom.minidom import Document

import re._compiler
from healthcare_rag.data.loaders.base_loader import Document
class TextCleaner:

    def __init__(self):
        self.page_number_pattern=re._compile(r'\n\s*Page\s+\d+\s*\n',re.IGNORECASE)
        self.header_pattern=re.compile(r'\n\s*Header\s*\n',re.IGNORECASE)
        self.footer_pattern=re.compile(r'\n\s*Footer\s*\n',re.IGNORECASE)
        self.url_pattern=re.compile(r'http[s]?://\s+')
        self.email_pattern=re.compile(r'\s+@\s+\.\s+')

    def clean(self,documents: List[Document])->List[Document]:
        cleaned_docs=[]
        for doc in documents:
            cleaned_content=self._clean_text(doc.content)
            cleaned_doc=Document(
                content=cleaned_content,
                metadata=doc.metadata,
                source=doc.source

            )
            cleaned_docs.append(cleaned_doc)
        return cleaned_docs

    def _clean_text(self, text: str)-> str:
        if not text:
            return ""
        text=self.page_number_pattern.sub('\n',text)
        text=self.header_pattern.sub('/n',text)
        text=self.footer_pattern.sub('/n',text)

        text=re.sub(r'\s+',' ', text)
        text=re.sub(r'\n\s*\n','\n\n',text)
        text=text.strip()

        return text
    def remove_empty_chunks(self,documents: List[Document])->List[Document] :

        return [doc for doc in documents if doc.content and len(doc.content.strip()) >50 ]       