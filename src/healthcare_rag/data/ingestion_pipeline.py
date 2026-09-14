from pathlib import Path
from typing import List, Union
from healthcare_rag.data.loaders.base_loader import Document
from healthcare_rag.data.loaders.pdf_loaders import PDFLoader
from healthcare_rag.data.loaders.json_loader import JSONLoader
from healthcare_rag.data.loaders.text_loader import TextLoader
from healthcare_rag.data.processors.text_cleaner import TextCleaner
from healthcare_rag.data.processors.chunker import TextChunker, Chunk
from healthcare_rag.config.settings import get_settings

class IngestionPipeline:
    def __init__(self):
        self.settings=get_settings()

        self.pdf_loader=PDFLoader()
        self.json_loader=JSONLoader(text_field="text",metadata_fields=["question","answer"])
        self.text_loader=TextLoader()
        self.cleaner=TextCleaner()
        self.chunker=TextChunker(
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap
        )
    def ingest(self,source:Union[str,Path])-> List[Chunk]:
        source_path=Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"source not found: {source}")

        if source_path.is_file():
            documents=self._load_file(source_path)
        else:
            documents=self._load_directory(source_path)
        print(f"Loaded {len(documents)} document from {source}")
        cleaned_docs=self.cleaner.clean(documents)
        cleaned_docs=self.cleaner.remove_empty_chunks(cleaned_docs)
        print(f"cleaned to {len(cleaned_docs)}documents")

        chunks=self.chunker.chunk(cleaned_docs)
        print(f"created {len(chunks)} chunks")
        return chunks

    def _load_file(self, file_path: Path) ->List[Document]:
        suffix=file_path.suffix.lower()

        if suffix=='.pdf':
            return self.pdf_loader.load(str(file_path))
        if suffix in ['.json','.jsonl']:
            return self.json_loader.load(str(file_path))
        if suffix in {".txt", ".md", ".rst", ".log"}:
            return self.text_loader.load(str(file_path))
        else:
            print(f"skipping unsupported file: {file_path}")
            return []
    def _load_directory(self,dir_path:Path)->List[Document]:
        documents=[]

        supported_extensions={'.pdf','.json','.jsonl','.txt','.md','.rst','.log'}

        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    docs=self._load_file(file_path)
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return documents
