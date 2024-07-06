import re
import threading
import nltk
from booleanmodel.enums import QuerySpecialSymbols
from nltk.stem import WordNetLemmatizer

class TextUtilities:
    _symbols = '\\'+'\\'.join(s.symbol for s in QuerySpecialSymbols if s != QuerySpecialSymbols.WORD_DELIMITER)
    _norm_pattern = re.compile(f'[^\\w\\s{_symbols}]')
    _norm_index_pattern = re.compile(r'[^\w\s]')
    nltk.download('wordnet', quiet = True)
    _lemmatizer = WordNetLemmatizer()

    @classmethod
    def normalize(cls, text: str) -> str:
        no_punctuation = cls._norm_pattern.sub('', text)
        lowercase = no_punctuation.lower()
        return lowercase 
    
    @classmethod
    def normalize_for_indexing(cls, text: str) -> str:
        norm = cls.normalize(text)
        norm = cls._norm_index_pattern.sub('', norm)
        return norm

    @classmethod
    def tokenize(cls, string: str) -> list[str]:
        string = cls.normalize_for_indexing(string)
        return string.split()
    
    @classmethod
    def lemmatize(cls, token: str) -> str:
        return cls._lemmatizer.lemmatize(token)
    


class AtomicCounter:
    def __init__(self, start: int = 0) -> None:
        self._value: int = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
        
    @property
    def value(self) -> int:
        return self._value
