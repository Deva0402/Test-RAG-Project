"""Json document loader.
handles json files and jsonl formats.
common for medical q&a dataset
"""



import json
from typing import List,Dict,Any
from pathlib import Path

from git import Optional
from healthcare_rag.data.loaders.base_loader import BaseLoader, Document

class JSONLoader(BaseLoader):
    def __init__(self,text_field: str="text",metadata_fields: Optional[List[str]]=None):
        self.text_field=text_field
        self.metadata_fields=metadata_fields or []

    def validate(self, source:str)->bool :
        path=Path(source)
        return path.exists() and path.suffix.lower() in ['.json','.jsonl']
    def load(self,source: str) ->List[Document]:
        if not self.validate(source):
            raise ValueError(f"invalid json source: {source}")
        path=Path(source)
        documents=[]

        if path.suffix.lower()=='.jsonl':
            with open(path,'r',encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        data=json.loads(line)
                        doc=self._parse_item(data,path,line_num)
                        if doc:
                            documents.append(doc)
        else:
            with open(path,'r',encoding='utf-8') as f:
                data=json.load(f)
                if isinstance(data,list):
                    for idx, item in enumerate(data):
                        doc=self._parse_item(item,path,idx)
                        if doc:
                            documents.append(doc)
                else:
                    doc=self._parse_item(data,path,0)
                    if doc:
                        documents.append(doc)
        return documents

    def _parse_item(self,item: Dict[str,Any],path:Path,index: int) ->Optional[Document]:
        if self.text_field not in item:
            return None

        text=item[self.text_field]
        if not text or not text.strip():
            return None
        metadata={
            "source_type":"json",
            "filename": path.name,
            "index":index,
        }
        for field in self.metadata_fields:
            if field in item:
                metadata[field]=item[field]

        return Document(content=text,metadata=metadata, source=str(path))                        
