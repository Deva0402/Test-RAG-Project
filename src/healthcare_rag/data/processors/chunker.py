from typing import List,Optional
from dataclasses import dataclass
import tiktoken
from healthcare_rag.data.loaders.base_loader import Document

@dataclass
class Chunk:
    content:str
    metadata:dict
    source:str
    chunk_index:int
    start_char: int
    end_char: int

class TextChunker:
    def __init__(
            self,
            chunk_size: int=512,
            chunk_overlap:int=64,
            encoding_name: str="cl100k_base"


    ):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.encoding=tiktoken.get_encoding(encoding_name)

    def chunk(self,documents: List[Document])->List[Chunk]:
        all_chunks=[]
        for doc in documents:
            chunks=self._chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _chunk_document(self,document: Document)-> List[Chunk]:
        if not document.content:
            return []

        tokens=self.encoding.encode(document.content)

        if len(tokens) <=self.chunk_size:
            return [Chunk(
                content=document.content,
                metadata=document.metadata,
                source=document.source,
                chunk_index=0,
                start_char=0,
                end_char=len(document.content)
                
            )]
        chunks=[]
        start=0
        chunk_index=0
        while start<len(tokens):
            end=min(start + self.chunk_size,len(tokens))
            chunk_tokens=tokens[start:end]

            chunk_text=self.encoding.decode(chunk_tokens)

            chunk_start_char=len(self.encoding.decode(tokens[:start]))
            chunk_end_char=len(self.encoding.decode(tokens[:end]))

            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunk_index": chunk_index,
                "chunk_token_count": len(chunk_tokens),

            })
            chunk_obj=Chunk(
                content=chunk_text,
                metadata=chunk_metadata,
                source=document.source,
                chunk_index=chunk_index,
                start_char=chunk_start_char,
                end_char=chunk_end_char
            )
            chunks.append(chunk_obj)

            start=end-self.chunk_overlap
            chunk_index+=1
        return chunks
    def estimate_token_count(self, text: str)-> int:
        tokens=self.encoding.encode(text)
        return len(tokens)    
        
