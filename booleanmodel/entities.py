from booleanmodel.data_structures import BooleanMap, HashMap, Trie
from booleanmodel.enums import QuerySpecialSymbols
from booleanmodel.utilities import TextUtilities, AtomicCounter
from booleanmodel.exceptions import ImpossibleMergeError, InvalidQueryException
from collections.abc import Iterator
from itertools import chain
from typing import Self, cast
from functools import total_ordering, reduce
import abc
import re

# Document class represents the document itself
# It can be subclassed to include richer content and features
# Override get_content_as_string() function
# to provide the system the fields of the document to have them indexed
class Document:
    _counter = AtomicCounter()

    def __init__(self, title: str, content) -> None:
        self.docID: int = self.__class__._counter.next()
        self.title: str = title
        self.content = content
    
    def __repr__(self) -> str:
        return self.title
    
    def __hash__(self) -> int:
        return hash(self.docID)
    
    def get_content_as_string(self) -> str:
        return str(self.content)

@total_ordering
class Posting:
    def __init__(self, docID: int, *positions: int) -> None:
        self.docID: int = docID
        self.term_positions: set[int] = set(positions)
        
    def merge(self, other: Self) -> Self:
        if (self.docID == other.docID):
            self.term_positions.update(other.term_positions)
        else:
            raise ImpossibleMergeError
        return self
    
    def union(self, other: Self) -> Self:
        if (self.docID == other.docID):
            return self.__class__(self.docID, *self.term_positions, *other.term_positions)
        else:
            raise ImpossibleMergeError

    def __eq__(self, other) -> bool:
        return self.docID == other.docID
    
    def __hash__(self) -> int:
        return hash(self.docID)
    
    def __gt__(self, other) -> bool:
        return self.docID > other.docID
    
    def __repr__(self) -> str:
        return str(self.docID)

class PostingList:
    def __init__(self) -> None:
        self._postings: BooleanMap[int, Posting] = HashMap()

    @classmethod
    def init_from(cls, docID: int, *term_positions: int) -> Self:
        pl = cls()
        pl._postings.add(docID, Posting(docID, *term_positions))
        return pl
    
    @classmethod
    def _from(cls, postings: BooleanMap[int, Posting]) -> Self:
        plist = cls()
        plist._postings = postings
        return plist    

    def __iter__(self) -> Iterator[int]:
        return iter(self._postings)
    
    def __getitem__(self, docId: int) -> Posting:
        return self._postings[docId]

    def union(self, other: Self) -> Self:
        return self.__class__._from(self._postings.union(other._postings))
    
    def intersection(self, other: Self) -> Self:
        return self.__class__._from(self._postings.intersection(other._postings))
    
    def merge(self, other: Self) -> Self:
       self._postings.merge(other._postings, Posting.merge)
       return self

    def get_docIDs(self) -> set[int]:
        return set(p for p in self._postings.keys())
    
    def __len__(self) -> int:
        return len(self._postings)

    def __repr__(self) -> str:
        return "[" + ", ".join((repr(p) for p in self._postings)) + "]"
    
class Term:
    def __init__(self, term: str, first_docID: int, *positions_in_first_doc: int) -> None:
        self.string: str = term
        self.posting_list: PostingList = PostingList.init_from(first_docID, *positions_in_first_doc)

    def merge(self, other: Self) -> Self:
        if (self == other):
            self.posting_list.merge(other.posting_list)
        return self        

    def __eq__(self, other) -> bool:
        return self.string == other.string
    
    def __gt__(self, other) -> bool:
        return self.string > other.string
    
    def __repr__(self) -> str:
        return f"{self.string}: {self.posting_list!r}"
    
    def __hash__(self) -> int:
        return hash(self.string)
    
class KGram:
    def __init__(self, kgram_str: str, *terms: Term) -> None:
        self.string: str = kgram_str
        self._terms: HashMap[str, Term] = HashMap({t.string: t for t in terms}, Term.merge)

    @staticmethod
    def build_kgrams_without_delimiter(term_str: str, k_length: int) -> tuple[str, ...]:
        return tuple(term_str[i:i+k_length] for i in range(len(term_str) - k_length + 1))

    @classmethod
    def build_kgrams_strings(cls, term_str: str, k_length: int, build_first_two_kgrams: bool = True) -> tuple[str, ...]:
        # build_first_two_kgrams e.g.
        # if True,  DRONE -> $DR, DRO, RON, ONE, NE$
        # if False, DRONE -> RON, ONE, NE$
        # where $ is start/end/delimiter of word symbol
        delimiter = QuerySpecialSymbols.WORD_DELIMITER.symbol
        delim_term = ''
        if (build_first_two_kgrams):
            delim_term = f'{delimiter}{term_str}{delimiter}'
        else:
            delim_term = f'{term_str[1:]}{delimiter}'

        return cls.build_kgrams_without_delimiter(delim_term, k_length)
    
    @classmethod
    def build_kgrams(cls, term: Term, k_length: int, build_first_two_kgrams: bool = True) -> tuple[Self, ...]:
        l = tuple(cls(kstr, term) for kstr in cls.build_kgrams_strings(term.string, k_length, build_first_two_kgrams))
        return l

    def merge(self, other: Self) -> Self:
        self._terms.merge(other._terms, Term.merge)
        return self
    
    def get_terms(self):
        return set(self._terms.values())
    
    def __repr__(self) -> str:
        return f'{self.string} -> [{", ".join(t.string for t in self._terms.values())}]'

class Index(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, term: str) -> PostingList:
        pass

    @abc.abstractmethod
    def wildcard_search(self, string: str) -> PostingList:
        pass

class InvertedIndex(Index):
    def __init__(self) -> None:        
        self.kgrams_length = 3
        self._build_first_two_kgrams = False
        self._dictionary: Trie[Term] = Trie()
        self._kgrams_dict: Trie[KGram] = Trie()

    @classmethod
    def from_corpus(cls, corpus: list[Document]) -> Self:
        index = cls()
        for doc in corpus:
            if (doc.docID % 1000 == 0):
                print(f"Processing document with ID: {doc.docID}...")
            tokens = TextUtilities.tokenize(doc.title + ' ' + doc.get_content_as_string())
            for position, token in enumerate(tokens):
                term = Term(token, doc.docID, position)
                index._dictionary.add(token, term, Term.merge)
                lem_token = TextUtilities.lemmatize(token)
                if (lem_token != token):
                    term = Term(lem_token, doc.docID, -1)
                    index._dictionary.add(lem_token, term, Term.merge)
                kgrams = KGram.build_kgrams(term, index.kgrams_length, index._build_first_two_kgrams)
                index._kgrams_dict.update({k.string: k for k in kgrams}, KGram.merge)
        print(f'\nIndexed {len(index._dictionary)} words and {len(index._kgrams_dict)} {index.kgrams_length}-grams')
        return index
    
    def __getitem__(self, term: str) -> PostingList:
        try:
            return self._dictionary[term].posting_list
        except KeyError:
            return PostingList()
        
    def wildcard_search(self, string: str) -> PostingList:
        wc = QuerySpecialSymbols.WILDCARD.symbol
        delim = QuerySpecialSymbols.WORD_DELIMITER.symbol
        spaced = re.compile(f'\\{wc}+').sub(f' {wc} ', string)
        tokens = spaced.split()
        pattern = ''.join(t if t != wc else f'\\w{wc}' for t in tokens)
        if (len(tokens) < 2):
            raise InvalidQueryException(f'Expected at least one word and one wildcard symbol "{QuerySpecialSymbols.WILDCARD.symbol}"')
        partial_sets_of_terms: list[set[Term]] = []
        first_part_terms: set[Term] = set()
        is_word_leading = tokens[0] != wc
        is_word_trailing = tokens[-1] != wc
        if (is_word_leading):
            terms_map = self._dictionary.items_with_prefix(tokens[0])
            first_part_terms = set(t for _,t in terms_map)
            if (len(tokens) == 2):
                posting_lists = (t.posting_list for t in first_part_terms)
                return reduce(PostingList.union, posting_lists)
            tokens = tokens[2:]
        # now tokens[0] == word1

        if (is_word_trailing):
            kgrams = KGram.build_kgrams_without_delimiter(f'{tokens[-1]}{delim}', self.kgrams_length)
            if (len(kgrams) > 0):
                kgrams_found = (self._kgrams_dict[k] for k in kgrams)
                partial_sets_of_terms.extend(map(KGram.get_terms, kgrams_found))                
            tokens = tokens[:-2]

        if (len(tokens) > 0):
            # now tokens is like: [word1, *, word2, *, word3, * ...]
            # now we have to do only trailing wildcard searches inside kgrams index
            words = tuple(filter(lambda t: t != wc, tokens))
            # generate a list of kgrams for each word
            kgrams_list = map(lambda w: KGram.build_kgrams_without_delimiter(w, self.kgrams_length), words)
            # flatten list of kgrams and get KGram objects
            kgrams_found = (self._kgrams_dict[k] for k in chain.from_iterable(kgrams_list))
            partial_sets_of_terms.extend(map(KGram.get_terms, kgrams_found))
            # for words shorter than a kgram we search them as prefixes of kgrams
            short_words = tuple(w for w in words if len(w) < self.kgrams_length)
            for w in short_words:
                # we get all kgrams that have prefix w
                kgs =  list(k for _,k in self._kgrams_dict.items_with_prefix(w))
                temp_list_set_terms = list(map(KGram.get_terms, kgs))
                # we do the union of all terms of all kgrams we got because all belongs to the same word w
                w_terms = reduce(set.union, temp_list_set_terms, cast(set[Term], set()))
                partial_sets_of_terms.append(w_terms)

        # intersecate kgrams results
        final_terms = reduce(set.intersection, partial_sets_of_terms)
        # and intersecate with first part terms if word leading
        if (is_word_leading):
            final_terms = final_terms.intersection(first_part_terms)

        # filter out results that still not match the pattern (due to words long less than a kgram)
        # and get posting lists
        filtered = (t.posting_list for t in final_terms if re.fullmatch(pattern, t.string) is not None)        
        return reduce(PostingList.union, filtered, PostingList())
