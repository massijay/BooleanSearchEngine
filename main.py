from functools import total_ordering, reduce
import abc
from  collections.abc import MutableSet, MutableMapping, Iterable, Mapping, ItemsView, Iterator, Callable, Hashable
import csv  # Import the csv module for CSV file parsing
import re  # Import the re module for regular expression operations
from typing import Any, Self, TypeVar, Protocol, Generic, Optional, cast
import threading
import sys
from enum import Enum


class Comparable(Protocol):    
    @abc.abstractmethod
    def __eq__(self, other, /) -> bool:
        ...
        
    @abc.abstractmethod
    def __lt__(self, other, /) -> bool:
        ...
        
    @abc.abstractmethod
    def __le__(self, other, /) -> bool:
        ...

    @abc.abstractmethod
    def __gt__(self, other, /) -> bool:
        ...    
        
    @abc.abstractmethod
    def __ge__(self, other, /) -> bool:
        ...

H = TypeVar("H", bound=Hashable)
S = TypeVar("S")
T = TypeVar("T")

#region sets

class BooleanMap(MutableMapping[S, T]):
        @abc.abstractmethod
        def __init__(self, mapping: Mapping[S,T] | None = None, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
            ...

        @abc.abstractmethod
        def union(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
            ...
        
        @abc.abstractmethod
        def intersection(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
            ...

        @abc.abstractmethod
        def merge(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
            ...

        @abc.abstractmethod
        def add(self, key: S, value: T, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
            ...
        
        @abc.abstractmethod
        def items(self) -> ItemsView[S, T]:
            ...

        def __repr__(self) -> str:
            return "{" + ", ".join((str(e) for e in self)) + "}"
        
class HashMap(BooleanMap[H, T]):
    def __init__(self, mapping: Mapping[H, T] | None = None, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
        self._dict: dict[H, T] = {}
        if (mapping is not None):
            for k,v in mapping.items():
                self.add(k, v, on_same_element_callback)

    def add(self, key: H, value: T, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
        if (key in self._dict):
            self._dict[key] = on_same_element_callback(self._dict[key], value)
        else:
            self._dict[key] = value

    def items(self) -> ItemsView[H, T]:
        return self._dict.items()
    
    def merge(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
        for k,v in other.items():
            self.add(k, v, on_same_element_callback)
        return self

    def union(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
        union = HashMap(self, on_same_element_callback)
        union.merge(other, on_same_element_callback)
        return union # type: ignore
    
    def intersection(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
        result = HashMap()
        for k,v in self.items():
            try:
                result.add(k, on_same_element_callback(v, other[k]))
            except KeyError:
                pass
        return result # type: ignore

    def __getitem__(self, key: H) -> T:
        return self._dict[key]
    
    def __setitem__(self, key: H, value: T) -> None:
        self.add(key, value)
    
    def __delitem__(self, key: H) -> None:
        del self._dict[key]

    def __iter__(self) -> Iterator[H]:
        return iter(self._dict)
    
    def __len__(self) -> int:
        return len(self._dict)
    
class TrieNode(Generic[T]):
    def __init__(self, value: Optional[T] = None) -> None:
        self.children: dict[str, TrieNode[T]] = {}
        self.value: Optional[T] = value

    def has_value(self) -> bool:
        return self.value is not None
    
    def has_children(self) -> bool:
        return len(self.children) > 0

class Trie(BooleanMap[str, T]):
    def __init__(self, mapping: Mapping[str, T] | None = None, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
        self._root: TrieNode[T] = TrieNode()
        self._size = 0
        if (mapping is not None):
            for k,v in mapping.items():
                self.add(k, v, on_same_element_callback)

    # TODO: docs
    # returns the part of the key already present in the tree
    #  and the leaf node
    def _find_prefix_node(self, key: str) -> tuple[str, TrieNode[T]]:
        node: TrieNode[T] = self._root
        for i in range(len(key)):
            if (key[i] in node.children):
                node = node.children[key[i]]
            else:
                return (key[:i], node)
        return (key, node)
    
    # TODO: docs
    # return the node where to operate when deleting a word
    # and the index incremented by 1 of the letter of the child that points to this node 
    # (0 if root node)
    # i.e. interesting letter index of the intersting node children
    def _find_start_of_last_subtree(self, string: str) -> tuple[int, TrieNode[T]]:
        next: TrieNode[T] = self._root
        node = next
        index = 0
        for i in range(len(string)):
            next = next.children[string[i]]
            if (len(next.children) > 1 or (len(next.children) > 0 and next.has_value())):
                # TODO: sistemare commenti
                index = i + 1
                node = next
        return (index, node)

    def add(self, key: str, value: T, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
        prefix, node = self._find_prefix_node(key)
        if (key == prefix):
            if (node.value is None):
                node.value = value
                self._size += 1
            else:
                node.value = on_same_element_callback(node.value, value)
            return        
        suffix = key[len(prefix):]
        next = TrieNode(value)
        # stop before the first letter of the suffix (iterating in reverse)
        for i in range(len(suffix) - 1, 0, -1):
            curr = TrieNode()
            curr.children[suffix[i]] = next
            next = curr
        node.children[suffix[0]] = next
        self._size += 1

    def update(self, mapping: Mapping[str, T], on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None: # type: ignore
        for k,v in mapping.items():
            self.add(k, v, on_same_element_callback)

    def get_if_present(self, key: str) -> Optional[T]:
        prefix, node = self._find_prefix_node(str(key))
        return node.value if prefix == key and node.has_value() else None

    def __contains__(self, key: object) -> bool:
        return self.get_if_present(str(key)) is not None
    
    def __getitem__(self, key: str) -> T:
        item = self.get_if_present(key)
        if (item is not None):
            return item
        raise KeyError
    
    def __setitem__(self, key: str, value: T) -> None:
        self.add(key, value)

    def __delitem__(self, key: str) -> None:
        index, node = self._find_start_of_last_subtree(key)
        if (key == key[:index]):
            # end of word => node has value
            if (not node.has_value()):
                raise KeyError
            node.value = None
            self._size -= 1
        elif (node.has_children()):
            # not end of word => node has children, otherwise the func above shouldn't have returned this node
            del node.children[key[index]]
            self._size -= 1
        else:
            raise KeyError
        
    def __iter__(self) -> Iterator[str]:
        return (k for k,_ in self.items())
    
    def _items_from_node(self, root_node: TrieNode[T]):
        def gen():   # -> Any helps to hide the incompatible type error
            chars: list[str] = []
            ch_iterators = [iter(root_node.children.items())]
            # items returned by the ch_iterator are unordered
            # => so this generator returns items in partial order
            while (len(ch_iterators) > 0):
                try:
                    char, node = next(ch_iterators[-1])
                    chars.append(char)
                    if (node.has_children()):
                        ch_iterators.append(iter(node.children.items()))
                        if (node.has_value()):
                            yield ("".join(chars), cast(T, node.value))
                    else: # doesn't have children => it has value
                        yield ("".join(chars), cast(T, node.value))
                        chars.pop()
                except StopIteration:
                    ch_iterators.pop()
                    if (len(chars) > 0):
                        chars.pop()
        return gen()
    
    def items_with_prefix(self, prefix: str):
        prefix_found, node = self._find_prefix_node(prefix)
        if (prefix != prefix_found):
            return ()
        root = TrieNode()
        root.children[""] = node
        return ((prefix + k, v) for k,v in self._items_from_node(root)) 
    
    def items(self): # type: ignore # items() shuold return a ItemsView (TODO)
        return self._items_from_node(self._root)
    
    def __len__(self) -> int:
        return self._size
    
    def merge(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
        for k,v in other.items():
            self.add(k, v, on_same_element_callback)
        return self
    
    def union(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
        union = Trie(self, on_same_element_callback)
        union.merge(other, on_same_element_callback)
        return union # type: ignore
    
    def intersection(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
        result = Trie()
        for k,v in self.items():
            vo = other.get_if_present(k)
            if (vo is not None):
                result.add(k, on_same_element_callback(v, vo))
        return result # type: ignore

#endregion sets

class ImpossibleMergeError(Exception):
    pass

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
        return repr(self.content)
    
#TODO Corpus class? with dict of documents?

@total_ordering
class Posting:
    def __init__(self, docID: int, *positions: int) -> None:
        self.docID: int = docID #TODO _docID or docID?
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
    def __init__(self) -> None: #TODO overload with first docID? with existing list?
        self.postings: BooleanMap[int, Posting] = HashMap() # TODO private 

    @classmethod
    def init_from(cls, docID: int, *term_positions: int) -> Self:
        pl = cls()
        pl.postings.add(docID, Posting(docID, *term_positions))
        return pl
    
    @classmethod
    def _from(cls, postings: BooleanMap[int, Posting]) -> Self:
        plist = cls()
        plist.postings = postings
        return plist    

    def __iter__(self) -> Iterator[int]:
        return iter(self.postings)
    
    def __getitem__(self, docId: int) -> Posting:
        return self.postings[docId]

    def union(self, other: Self) -> Self:
        return self.__class__._from(self.postings.union(other.postings))
    
    def intersection(self, other: Self) -> Self:
        return self.__class__._from(self.postings.intersection(other.postings))
    
    def merge(self, other: Self) -> Self:
       self.postings.merge(other.postings, Posting.merge)
       return self

    def get_docIDs(self) -> set[int]:
        return set(p for p in self.postings.keys())
    
    def __len__(self) -> int:
        return len(self.postings)

    def __repr__(self) -> str:
        return "[" + ", ".join((repr(p) for p in self.postings)) + "]"
    
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
        # self._terms: set[Term] = {*terms} #TODO public?
        self._terms: HashMap[str, Term] = HashMap({t.string: t for t in terms}, Term.merge)
    
    @classmethod
    def build_kgrams(cls, term: Term, k_length: int, build_first_two_kgrams: bool = True) -> list[Self]:
        # build_leading_kgrams e.g.
        # if True,  DRONE -> $DR, DRO, RON, ONE, NE$
        # if False, DRONE -> RON, ONE, NE$
        # where $ is start/end/delimiter of word symbol
        kgrams: list[Self] = []
        delimiter = QuerySpecialSymbols.WORD_DELIMITER.symbol

        term_str = ''
        if (build_first_two_kgrams):
            term_str = f'{delimiter}{term.string}{delimiter}'
        else:
            term_str = f'{term.string[1:]}{delimiter}'

        for i in range(len(term_str) - k_length + 1):
            kgram = cls(term_str[i:i+k_length], term)
            kgrams.append(kgram)

        return kgrams

    def merge(self, other: Self) -> Self:
        self._terms.merge(other._terms, Term.merge)
        return self

#TODO improve
def normalize(text: str) -> str:
    no_punctuation = re.sub(r'[^\w^\s^-]','',text)
    lowercase = no_punctuation.lower()
    return lowercase

#TODO improve
def tokenize(document: Document) -> list[str]:
    title = normalize(document.title)
    content = normalize(document.get_content_as_string())
    return title.split() + content.split()

class Index(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, term: str) -> PostingList:
        pass

    @abc.abstractmethod
    def wildcard_search(self, string: str) -> PostingList:
        pass

class InvertedIndex(Index):
    def __init__(self) -> None:
        # simple list: 1.33 GB of RAM -> 201799 terms
        #self._dictionary: list[Term] = []

        # HashMap: 1.34 GB
        #self._dictionary: HashMap[str, Term] = HashMap()

        # Trie: 1.49 GB
        # 2 Tries (dict + kgrams dict): 2.15 GB, indexing time: 4 minutes
        self._dictionary: Trie[Term] = Trie()
        self._kgrams_dict: Trie[KGram] = Trie()

    
    @classmethod
    def from_corpus(cls, corpus: dict[int, Document]) -> Self:        
        index = cls()
        for doc in corpus.values():
            if (doc.docID % 1000 == 0):
                print(f"Processing document with ID: {doc.docID}...")
            tokens = tokenize(doc)
            for position, token in enumerate(tokens):
                term = Term(token, doc.docID, position)
                index._dictionary.add(token, term, Term.merge)
                kgrams = KGram.build_kgrams(term, 3, False)
                index._kgrams_dict.update({k.string: k for k in kgrams}, KGram.merge)
        return index
    
    def __getitem__(self, term: str) -> PostingList:
        try:
            return self._dictionary[term].posting_list
        except KeyError:
            return PostingList()
        
    def wildcard_search(self, string: str) -> PostingList: #TODO implement
        if (WildcardType.TRAILING.matches(string)):
            word = string[:-1]
            terms_map = self._dictionary.items_with_prefix(word)
            posting_lists = (t.posting_list for _,t in terms_map)
            pl = reduce(PostingList.merge, posting_lists)
            return pl
        if (WildcardType.LEADING.matches(string)):
            word = string[1:]

        if (WildcardType.GENERAL.matches(string)):
            words = string.split(f'{QuerySpecialSymbols.WILDCARD.symbol}')
        raise Exception #TODO invalid wildcard pattern
        
class QuerySpecialSymbols(Enum):
    AND = '&'
    OR = '|'
    MINUS = '-'
    OPEN_PARENTHESIS = '('
    CLOSED_PARENTHESIS = ')'
    QUOTE = '"'
    WILDCARD = '*'
    WORD_DELIMITER = '\x03' # ETX hex code
    
    @property
    def symbol(self) -> str:
        return self.value
    
class OperatorType(Enum):
    AND = QuerySpecialSymbols.AND.symbol
    OR = QuerySpecialSymbols.OR.symbol
    MINUS = QuerySpecialSymbols.MINUS.symbol
    WILDCARD = QuerySpecialSymbols.WILDCARD.symbol
    
    @property
    def symbol(self) -> str:
        return self.value
    
class TokenType(Enum):
    WORD = re.compile('\\w+') #TODO clean up query and terms before building index (normalize)
    # WILDCARD_WORD = re.compile('[\\w\\*]*(?:(?:\\w\\*)|(?:\\*\\w))[\\w\\*]*') #TODO remove duplicate symbols in query(?)
    WILDCARD_WORD = re.compile(f'[\\w\\{QuerySpecialSymbols.WILDCARD.symbol}]*(?:(?:\\w\\{QuerySpecialSymbols.WILDCARD.symbol})|(?:\\{QuerySpecialSymbols.WILDCARD.symbol}\\w))[\\w\\{QuerySpecialSymbols.WILDCARD.symbol}]*')
    OPERATOR = re.compile(f'[{QuerySpecialSymbols.AND.symbol}{QuerySpecialSymbols.OR.symbol}{QuerySpecialSymbols.MINUS.symbol}]')
    OPEN_PARENTHESIS = re.compile(f'\\{QuerySpecialSymbols.OPEN_PARENTHESIS.symbol}')
    CLOSED_PARENTHESIS = re.compile(f'\\{QuerySpecialSymbols.CLOSED_PARENTHESIS.symbol}')
    QUOTE = re.compile(QuerySpecialSymbols.QUOTE.symbol)

    def matches(self, string: str) -> bool:
        m = re.fullmatch(self.value, string)
        return m is not None

class WildcardType(Enum):
    TRAILING = re.compile(f'^\\w+\\{QuerySpecialSymbols.WILDCARD.symbol}$')
    LEADING = re.compile(f'^\\{QuerySpecialSymbols.WILDCARD.symbol}\\w+$')
    GENERAL = TokenType.WILDCARD_WORD.value

    def matches(self, string: str) -> bool:
        m = re.fullmatch(self.value, string)
        return m is not None

class QueryNode(abc.ABC):
    @abc.abstractmethod
    def query(self, index: Index) -> set[int]:
        pass

class Token(QueryNode): #TODO word of the query, name correct?
    def __init__(self, string: str) -> None:
        self._string = string

    def query(self, index: Index) -> set[int]:
        return index[self._string].get_docIDs()
    
class Prefix(QueryNode):
    def __init__(self, string: str) -> None:
        self._string = string

    def query(self, index: Index) -> set[int]:
        return index[self._string].get_docIDs()
    
class WildcardToken(Token):
    def __init__(self, string: str) -> None:
        super().__init__(string)
    
    def query(self, index: Index) -> set[int]:
        return index.wildcard_search(self._string).get_docIDs()

class Phrase(QueryNode):
    def __init__(self, ordered_tokens: list[str]) -> None:
        self._terms = ordered_tokens

    def query(self, index: Index) -> set[int]:
        if (len(self._terms) == 0):
            return set()
        if (len(self._terms) == 1):
            return index[self._terms[0]].get_docIDs()
        # search the entire phrase
        # get the posting list of each term
        posting_lists: list[PostingList] = []
        for t in self._terms:
            pl = index[t]
            if (len(pl) > 0):
                posting_lists.append(pl)
            else:
                return set()
        # get a set of the common docIDs
        docIDs_list = map(PostingList.get_docIDs, posting_lists)
        common_docIDs = reduce(lambda s1, s2: s1.intersection(s2), docIDs_list)
        phrase_docIDs: set[int] = set()
        # for each doc ID
        for docID in common_docIDs: # for 1
            phrase_found = False
            # get the positions of the first term
            first_term_positions = posting_lists[0][docID].term_positions
            for p in first_term_positions: # for 2
                # for each of the other terms:
                phrase_found = True
                for i in range(1, len(self._terms)): # for 3
                    ith_term_positions = posting_lists[i][docID].term_positions
                    # check if after i skips after the first term there is the i-th term,
                    #  if not, stop checking the next terms positions (for this first term position)
                    if ((p + i) not in ith_term_positions):
                        # phrase is not complete
                        phrase_found = False 
                        break # for 3
                # if the phrase is present add this doc ID to the result
                if (phrase_found):
                    phrase_docIDs.add(docID)
                    # no need to evaluate other positions of the terms for this document,
                    # phrase already found in it
                    break # for 2
        return phrase_docIDs

class Operator(QueryNode):
    def __init__(self, symbol: str, left_operand: QueryNode, right_operand: QueryNode) -> None:
        self._left_operand = left_operand
        self._right_operand = right_operand
        self._type = OperatorType(symbol)

    def query(self, index: Index) -> set[int]:
        left_docIDs = self._left_operand.query(index)
        right_docIDs = self._right_operand.query(index)
        match self._type:
            case OperatorType.AND:
                return left_docIDs.intersection(right_docIDs)
            case OperatorType.OR:
                return left_docIDs.union(right_docIDs)
            case OperatorType.MINUS:
                return left_docIDs.difference(right_docIDs)
            case _:
                raise Exception # TODO
            
class Query:
    @staticmethod
    def _are_quotes_balanced(string: str) -> bool:
        quotes = filter(lambda x: x == '"', string)
        num = sum(1 for _ in quotes)
        return num % 2 == 0
    
    @staticmethod
    def _are_parenthesis_balanced(string: str) -> bool:
        pars = filter(lambda x: x == "(" or x == ")", string)
        num = reduce(lambda sum, char: sum + (1 if char == "(" else -1), pars, 0)
        return num == 0

    @staticmethod
    def _space_symbols(string: str) -> str:
        # put spaces around all special symbols
        for sym in QuerySpecialSymbols:
            string = string.replace(sym.symbol, f" {sym.symbol} ")
        
        # remove spaces before and after wildcards if a word is respectively before or after
        # this is done in two steps because regex doesn't execute twice on overlapping matches
        w1 = re.compile(f'(?:(\\w+)\\s*)?\\{QuerySpecialSymbols.WILDCARD.symbol}')
        w2 = re.compile(f'\\{QuerySpecialSymbols.WILDCARD.symbol}(?:\\s*(\\w+))?')
        string = w1.sub(f'\\1{QuerySpecialSymbols.WILDCARD.symbol}', string)
        string = w2.sub(f'{QuerySpecialSymbols.WILDCARD.symbol}\\1', string)
        return string
    
    @staticmethod
    def _is_word(string: str) -> bool:
        x = re.search("[a-zA-Z0-9]+", string) #TODO
        return x is not None

    @staticmethod
    def _replace_spaces_with_ands(words: list[str]) -> list[str]:
        if (len(words) == 0):
            return [] 
        correct: list[str] = []
        for w in words[:-1]:
            correct += [w, '&' , '(']
        correct += [words[-1]]
        correct += [')'] * (len(words) - 1)
        return correct
    
    @classmethod
    def _string_query_to_list(cls, query_string: str) -> list[str]:
        s = cls._space_symbols(query_string)
        full_list = s.split()
        nonempty_list = filter(lambda s: len(s) != 0, full_list)
        return ['(', *nonempty_list, ')']
    
    @classmethod
    def _fix_implicit_ands(cls, words: list[str]) -> list[str]:
        i = 0
        start = -1
        is_phrase = False
        while (i < len(words)):
            if (words[i] == '"'):
                is_phrase = not is_phrase
                i += 1
                continue
            if (is_phrase):
                i += 1
                continue
            if (cls._is_word(words[i])):
                if (start == -1):
                    start = i
                i += 1
                continue
            if (start != -1):
                before = words[:start]
                new_list = cls._replace_spaces_with_ands(words[start:i])
                words = before + new_list + words[i:]
                i = start + len(new_list)
                start = -1
                continue
            i += 1
        return words
    
    def __init__(self, query_string: str) -> None:
        cls = self.__class__
        if (not cls._are_quotes_balanced(query_string)):
            raise Exception #TODO: create exception
        if (not cls._are_parenthesis_balanced(query_string)):
            raise Exception #TODO
        # TODO: tokenization, normalization
        spaced = cls._space_symbols(query_string)
        temp_list = cls._string_query_to_list(spaced)
        self._query_list = cls._fix_implicit_ands(temp_list)
        self._i = 0
        self._root = self._parse()

    def _parse(self) -> QueryNode:
        ql = self._query_list

        if (TokenType.WORD.matches(ql[self._i])):
            node = Token(ql[self._i])
            self._i += 1
            return node
        
        if (TokenType.WILDCARD_WORD.matches(ql[self._i])):
            node = WildcardToken(ql[self._i])
            self._i += 1
            return node
        
        if (TokenType.QUOTE.matches(ql[self._i])):
            start = self._i + 1
            end = start
            while (TokenType.WORD.matches(ql[end])):
                end += 1
            if (not TokenType.QUOTE.matches(ql[end])):
                raise Exception # TODO Special symbol found before phrase end, expected "
            node = Phrase(ql[start:end])
            self._i = end + 1
            return node
        
        if (not TokenType.OPEN_PARENTHESIS.matches(ql[self._i])):
            raise Exception # TODO Expected "("
        self._i += 1
        left = self._parse()

        if (TokenType.CLOSED_PARENTHESIS.matches(ql[self._i])):
            self._i += 1
            return left

        if (not TokenType.OPERATOR.matches(ql[self._i])):
            raise Exception # TODO Expected valid OPERATOR symbol or ")"
        op_symbol = ql[self._i]
        self._i += 1
        right = self._parse()

        if (not TokenType.CLOSED_PARENTHESIS.matches(ql[self._i])):
            raise Exception # TODO Expected ")"
        self._i += 1
        return Operator(op_symbol, left, right)

    def execute(self, index: Index) -> set[int]:
        return self._root.query(index)

class BooleanIRSystem:
    def __init__(self, corpus: dict[int, Document], index: InvertedIndex) -> None:
        self._corpus: dict[int, Document] = corpus
        self._index: InvertedIndex = index

    @classmethod
    def from_corpus(cls, corpus: dict[int, Document]) -> Self:
        index = InvertedIndex.from_corpus(corpus)
        return cls(corpus, index)
    
    def query(self, query_string: str) -> list[Document]:
        query = Query(query_string)
        docIDs = query.execute(self._index)
        return [self._corpus[docID] for docID in docIDs]
        
def parse_corpus() -> dict[int, Document]:
    file_movie_names = "documents/movie.metadata.tsv"
    file_movie_plots = "documents/plot_summaries.txt"
    movies_dict = {}
    corpus: dict[int, Document] = {}

    with open(file_movie_names, 'r', encoding="utf8") as file:
        metadata = csv.reader(file, delimiter='\t')
        for row in metadata:
            # row[0] movie id
            # row[2] movie name
            movies_dict[row[0]] = row[2]
    
    with open(file_movie_plots, 'r', encoding="utf8") as file:
        plots = csv.reader(file, delimiter='\t')
        for plot in plots:
            try:
                # plot[0] movie id
                # plot[1] plot
                #doc = Document(movies_dict[plot[0]], plot[1]) # REMOVED PLOT FOR DEBUG
                doc = Document(movies_dict[plot[0]], "")
                corpus[doc.docID] = doc
            except KeyError:
                #TODO
                pass
    return corpus

corpus = parse_corpus()
ir = BooleanIRSystem.from_corpus(corpus)
print("ready!\n")
exit = False
while (not exit):
    q = input()
    if (q != "$$"):
        result = ir.query(q)
        print()
        for r in result:
            print(r)
        print()
    else:
        exit = True

        


