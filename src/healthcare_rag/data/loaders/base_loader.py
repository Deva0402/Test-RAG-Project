"""abstract base class for document loaders.
all loaders inherit from this to ensure consistent interface.
"""

from abc import ABC, abstractmethod
from importlib import metadata
from typing import List,Dict,Any
from dataclasses import dataclass

from numpy import source

@dataclass
class Document:
    """Represents a loaded document with metadata."""
    content: str
    metadata: Dict[str,Any]
    source:str

class Baseloader(ABC):
    """Abstract base class for document loaders.
    All loaders must implement:
    -load(): load documents from a source
    -validate(): check if source is valid
    """
    @abstractmethod
    def load(self,source: str) -> List[Document]:
        """
        Load documents from the given source.
        
        Args:
        Source: Path to file,URL,or other identifier
        
        Returns:
        List of document objects
        """
        pass
    @abstractmethod
    def validate(self,source: str) -> bool:
        """
        validate that the source can be loaded.
        
        Args:
            sorce: path to file , URL, or other identifier
        Returns:
            true if valid, False otherwise
                    """
        pass    
