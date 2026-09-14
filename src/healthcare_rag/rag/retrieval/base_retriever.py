from abc import ABC, abstractmethod
from typing import List,Dict,Any

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,query: str,
        top_k: int=5,
        **kwargs
    ) -> List[Dict[str,Any]]:
        pass