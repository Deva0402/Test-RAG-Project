from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseEmbedder(ABC):

    @abstractmethod
    def embed(self,text:str)->List[float]:
        pass

    @abstractmethod
    def embed_batch(self,texts: List[str])-> List[List[float]]:
        pass

    @property
    @abstractmethod
    def dimensions(self)-> int:
        pass
