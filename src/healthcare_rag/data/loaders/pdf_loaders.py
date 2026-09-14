"""PDF document loader using pypdf.
Extracts text from PDF files with basic metadata.
"""

from importlib import metadata
from typing import List
from pathlib import Path
from pypdf import PdfReader
from healthcare_rag.data.loaders.base_loader import Baseloader, Document

class PDFLoader(Baseloader):
    """
    Load text content from PDF files.
    Suitable for medical guidelines, research papers, clinical documents.
    """

    def validate(self,source:str) -> bool:
        """check if source is a valid PDF file."""
        path=Path(source)
        return path.exists() and path.suffix.lower()=='.pdf'
    
    def load(self,source: str) -> List[Document]:
        """Load text from PDF file.
        Args: 
        Source: path to PDF file
        
        Returns:
        List with one Document containing all PDF text
        """
        if not self.validate(source):
            raise ValueError(f"Invalid PDF source: {source}")
        path=Path(source)
        reader=PdfReader(path)

        text_parts=[]
        for page_num,page in enumerate(reader.pages):
            text=page.extract_text()
            if text.strip():
                text_parts.append(text)

        full_text ="\n\n".join(text_parts)
        metadata={
            "source_type":"pdf",
            "filename":path.name,
            "total_pages":len(reader.pages),
            "file_size":path.stat().st_size,
        }

        return [Document(content=full_text,metadata=metadata,source=str(path))]        