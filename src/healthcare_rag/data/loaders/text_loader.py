from typing import List
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from sympy import content
from healthcare_rag.data.loaders.base_loader import Baseloader, Document

class TextLoader(Baseloader):
    SUPPORTED_EXTENSIONS={'.txt','.md','.rst','.log'}
    def validate(self,source:str)-> bool:
        path=Path(source)
        return path.exists() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self,source:str)->List[Document]:
        if not self.validate(source):
            raise ValueError(f"Invalid text source: {source}")
        path=Path(source)

        with open(path,'r',encoding='utf-8') as f:
            content=f.read()

        metadata={
            "source_type": "text",
            "filename": path.name,
            "file_size":path.stat().st.size,
        } 
        return [Document(content=content,metadata=metadata,source=str(path))]