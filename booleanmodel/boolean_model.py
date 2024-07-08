from booleanmodel.entities import Document, InvertedIndex
from booleanmodel.query import Query
from typing import Self, cast
import pickle

class BooleanIRSystem:
    def __init__(self, corpus: dict[int, Document], index: InvertedIndex) -> None:
        self._corpus: dict[int, Document] = corpus
        self._index: InvertedIndex = index

    def save_to_file(self, directory_path: str, index_file_name = "index.pickle", corpus_file_name = "corpus.pickle"):
        self._index.save_to_file(f'{directory_path}/{index_file_name}')
        with open(f'{directory_path}/{corpus_file_name}', 'wb') as file:
            pickle.dump(self._corpus, file)

    @classmethod
    def load_from_file(cls, directory_path: str, index_file_name = "index.pickle", corpus_file_name = "corpus.pickle") -> Self:
        index = InvertedIndex.load_from_file(f'{directory_path}/{index_file_name}')
        corpus: dict[int, Document]
        with open(f'{directory_path}/{corpus_file_name}', 'rb') as file:
            data = pickle.load(file)
            corpus = cast(dict[int, Document], data)
        ir = cls(corpus, index)
        return ir

    @classmethod
    def from_corpus(cls, corpus: list[Document]) -> Self:
        index = InvertedIndex.from_corpus(corpus)
        return cls({d.docID:d for d in corpus}, index)
    
    def query(self, query_string: str) -> list[Document]:
        query = Query(query_string)
        docIDs = query.execute(self._index)
        return [self._corpus[docID] for docID in docIDs]
        