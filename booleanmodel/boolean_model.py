from booleanmodel.entities import Document, InvertedIndex
from booleanmodel.query import Query
from typing import Self

class BooleanIRSystem:
    def __init__(self, corpus: dict[int, Document], index: InvertedIndex) -> None:
        self._corpus: dict[int, Document] = corpus
        self._index: InvertedIndex = index

    @classmethod
    def from_corpus(cls, corpus: list[Document]) -> Self:
        index = InvertedIndex.from_corpus(corpus)
        return cls({d.docID:d for d in corpus}, index)
    
    def query(self, query_string: str) -> list[Document]:
        query = Query(query_string)
        docIDs = query.execute(self._index)
        return [self._corpus[docID] for docID in docIDs]
        