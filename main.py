from functools import total_ordering, reduce
import abc
from  collections.abc import MutableSet, MutableMapping, Iterable, Mapping, ItemsView, Iterator, Callable, Hashable
import csv  # Import the csv module for CSV file parsing
import re  # Import the re module for regular expression operations
from typing import Any, Self, TypeVar, Protocol, Generic, Optional, cast
import threading
import sys

#region sets

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
T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U", bound=Comparable)

class BooleanSet(MutableSet[T]):
        @abc.abstractmethod
        def __init__(self, iterable: Iterable) -> None:
            ...

        @abc.abstractmethod
        def union(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
            ...
        
        @abc.abstractmethod
        def intersection(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
            ...

        @abc.abstractmethod
        def merge(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
            ...

        def __repr__(self) -> str:
            return "{" + ", ".join((str(e) for e in self)) + "}"

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
        def merge(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
            ...

        @abc.abstractmethod
        def add(self, key: S, value: T, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
            ...
        
        @abc.abstractmethod
        def items(self) -> ItemsView[S, T]:
            ...

        def __repr__(self) -> str:
            return "{" + ", ".join((str(e) for e in self)) + "}"
        

class HashMap(BooleanMap[H, T]): # TODO: H is Hashable
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
    
    def merge(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
        for k,v in other.items():
            self.add(k, v, on_same_element_callback)

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
                # caso -culo culetto (nodo dopo l ha 2 figli)
                # caso -culone culo (nodo dopo "o" che ha value ed è anche di un altra parola (non è l'ultima lettera di key))
                #  (se nodo dopo "o" è anche di un altra parola allora uguale a caso sotto)
                # caso -culo culone (nodo dopo "o" deve perdere value ma non i fligli)
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
    
    def merge(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
        for k,v in other.items():
            self.add(k, v, on_same_element_callback)
    
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

class ListSet(BooleanSet[U]):
    def __init__(self, iterable: Iterable = []) -> None:
        self._list: list[U] = sorted(iterable)
            
    def __iter__(self) -> Iterator:
        return iter(self._list)

    def __contains__(self, x) -> bool:
        return x in self._list
    
    def __len__(self) -> int:
        return len(self._list)
    
    @classmethod
    def _from_sorted_list(cls, sorted_list: list[U]) -> Self:
        sl = cls()
        sl._list = sorted_list
        return sl
    
    def add(self, value: U) -> None:
        for i in range(len(self._list)):
            if (self._list[i] == value):
                self._list[i] = value
                return
            elif (self._list[i] > value):
                self._list.insert(i, value)
                return
        self._list.append(value)

    def discard(self, value: Any) -> None:
        for i in range(len(self._list)):
            if (self._list[i] == value):
                del self._list[i]
                return

    def union(self, other: Self, on_same_element_callback: Callable[[U, U], U] = lambda x, y: x) -> Self:
        result: list[U] = []
        i = j = 0
        while (i < len(self._list) and j < len(other._list)):
            if (self._list[i] < other._list[j]):
                result.append(self._list[i])
                i += 1
            elif (self._list[i] == other._list[j]):
                result.append(on_same_element_callback(self._list[i], other._list[j]))
                i += 1
                j += 1
            else:
                result.append(other._list[j])
                j += 1
        return self.__class__._from_sorted_list(result + self._list[i:] + other._list[j:])
    
    def intersection(self, other: Self, on_same_element_callback: Callable[[U, U], U] = lambda x, y: x) -> Self:
        result: list[U] = []
        i = j = 0
        while (i < len(self._list) and j < len(other._list)):
            if (self._list[i] == other._list[j]):
                result.append(on_same_element_callback(self._list[i], other._list[j]))
                i += 1
                j += 1
            elif (self._list[i] < other._list[j]):
                i += 1
            else:
                j += 1
        return self.__class__._from_sorted_list(result)
    
    def merge(self, other: Self, on_same_element_callback: Callable[[U, U], U] = lambda x, y: x) -> None:
        # one of the two lists is empty
        if (len(other._list) == 0):
            return
        if (len(self._list) == 0):
            self._list = other._list
            return
        
        # the self list elements are smaller than the others and
        #  they have an element in common
        if (self._list[-1] == other._list[0]):
            self._list[-1] = on_same_element_callback(self._list[-1], other._list[0])
            self._list += other._list[1:]
            return
        #  they haven't
        if (self._list[-1] < other._list[0]):
            self._list += other._list
            return
        
        # same as above but self and other lists are swapped
        if (other._list[-1] == self._list[0]):
            m_el = on_same_element_callback(other._list[-1], self._list[0])
            self._list = other._list[:-1] + [m_el] + self._list[1:]
            return
        if (other._list[-1] < self._list[0]):
            self._list = other._list + self._list
            return
        
        # the two lists are overlapping
        self._list = self.union(other)._list
            
class HashSet(BooleanSet[U]): # TODO: update all code from tests
    def __init__(self, iterable: Iterable[U] = []) -> None: # TODO: remove ALL mutable objects as default params 
        self._dict: dict[int, U] = {hash(e): e for e in iterable}
    
    def __iter__(self) -> Iterator:
        return iter(self._dict.values())
    
    def __contains__(self, x: object) -> bool:
        return hash(x) in self._dict
    
    def __len__(self) -> int:
        return len(self._dict)    
    
    @classmethod
    def _from_dict(cls, dict: dict[int, U]) -> Self:
        sl = cls()
        sl._dict = dict
        return sl
    
    def add(self, value: Any) -> None:
        self._dict[hash(value)] = value

    def discard(self, value: Any) -> None:
        try:
            del self._dict[hash(value)]
        except KeyError:
            pass

    def union(self, other: Self, on_same_element_callback: Callable[[Any, Any], Any] = lambda x, y: x) -> Self:
        result = self._dict.copy() if len(self._dict) > len(other._dict) else other._dict.copy()
        small = other._dict if len(other._dict) < len(self._dict) else self._dict
        for k in small:
            if (k in result):
                result[k] = on_same_element_callback(result[k], small[k])
            else:
                result[k] = small[k]
        return self.__class__._from_dict(result)
    
    def intersection(self, other: Self, on_same_element_callback: Callable[[U, U], U] = lambda x, y: x) -> Self:
        big = self._dict if len(self._dict) > len(other._dict) else other._dict
        small = other._dict if len(other._dict) < len(self._dict) else self._dict
        result = {k: on_same_element_callback(big[k], small[k]) for k in small if k in big}
        return self.__class__._from_dict(result)
    
    def merge(self, other: Self, on_same_element_callback: Callable[[U, U], U] = lambda x, y: x) -> None:
        for k in other._dict:
            if (k in self._dict):
                self._dict[k] = on_same_element_callback(self._dict[k], other._dict[k])
            else:
                self._dict[k] = other._dict[k]

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
        self.term_positions: list[int] = sorted(positions) # TODO can be a set? 

    def union(self, other: Self) -> Self:
        if (self.docID == other.docID):
            return self.__class__(self.docID, *self.term_positions, *other.term_positions)
        else:
            raise ImpossibleMergeError
        
    def merge(self, other: Self) -> None:
        self.term_positions = self.union(other).term_positions

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
    
    def merge(self, other: Self) -> None:
       self.postings.merge(other.postings, Posting.union)

    def get_docIDs(self) -> set[int]:
        return set(p for p in self.postings.keys())
    
    def __len__(self) -> int:
        return len(self.postings)

    def __repr__(self) -> str:
        return "[" + ", ".join((repr(p) for p in self.postings)) + "]"
    
    
# class PostingList:
#     def __init__(self) -> None: #TODO overload with first docID? with existing list?
#         self.postings: set[Posting] = set() #TODO: skiplist?

#     @classmethod
#     def _from_set(cls, postings: set[Posting]) -> Self:
#         plist = cls()
#         plist.postings = postings
#         return plist

#     @classmethod
#     def init_from(cls, docID: int, *term_positions: int) -> Self:
#         pl = cls()
#         pl.postings.add(Posting(docID, *term_positions))
#         return pl

#     def union(self, other: Self) -> Self:
#         return self.__class__._from_set(self.postings.union(other.postings))
    
#     def intersection(self, other: Self) -> Self:
#         return self.__class__._from_set(self.postings.intersection(other.postings))
    
#     def merge(self, other: Self) -> None:
#        self.postings.update(other.postings)

#     def __repr__(self) -> str:
#         return ", ".join((repr(p) for p in self.postings))
    
# class PostingList:
#     def __init__(self) -> None: #TODO overload with first docID? with existing list?
#         self.postings: list[Posting] = [] #TODO: skiplist?

#     @classmethod
#     def _from_list(cls, postings: list[Posting]) -> Self:
#         plist = cls()
#         plist.postings = postings
#         return plist

#     @classmethod
#     def init_from(cls, docID: int, *term_positions: int) -> Self:
#         pl = cls()
#         pl.postings = [Posting(docID, *term_positions)]
#         return pl

#     def union(self, other: Self) -> Self:
#         result: list[Posting] = []
#         i = j = 0
#         while (i < len(self.postings) and j < len(other.postings)):
#             if (self.postings[i] < other.postings[j]):
#                 result.append(self.postings[i])
#                 i += 1
#             elif (self.postings[i] == other.postings[j]):
#                 result.append(self.postings[i].union(other.postings[j]))
#                 i += 1
#                 j += 1
#             else:
#                 result.append(other.postings[j])
#                 j += 1
#         return self.__class__._from_list(result + self.postings[i:] + other.postings[j:])
    
#     def intersection(self, other: Self) -> Self:
#         result: list[Posting] = []
#         i = j = 0
#         while (i < len(self.postings) and j < len(other.postings)):
#             if (self.postings[i] == other.postings[j]):
#                 result.append(self.postings[i])
#                 i += 1
#                 j += 1
#             elif (self.postings[i] < other.postings[j]):
#                 i += 1
#             else:
#                 j += 1
#         return self.__class__._from_list(result)

    
#     def merge(self, other: Self) -> None:
#         self.postings = self.union(other).postings

#     def __repr__(self) -> str:
#         return ", ".join((repr(p) for p in self.postings))
    
class Term:
    def __init__(self, term: str, first_docID: int, *positions_in_first_doc: int) -> None:
        self.term: str = term
        self.posting_list: PostingList = PostingList.init_from(first_docID, *positions_in_first_doc)

    def merge(self, other: Self) -> Self:
        if (self == other):
            self.posting_list.merge(other.posting_list)
        return self
        

    def __eq__(self, other) -> bool:
        return self.term == other.term
    
    def __gt__(self, other) -> bool:
        return self.term > other.term
    
    def __repr__(self) -> str:
        return f"{self.term}: {self.posting_list!r}"
    
    def __hash__(self) -> int:
        return hash(self.term)
    
class KGram:
    def __init__(self, kgram: str, *terms: Term) -> None:
        self.kgram: str = kgram
        self._terms : set[Term] = {*terms} #TODO public?
    
    @classmethod
    def get_kgrams(cls, term: Term) -> list[Self]:
        #TODO build kgrams from term
        return []
    
    def add_terms(self, *terms: Term) -> None:
        self._terms.update(terms)


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

class InvertedIndex(Index):
    def __init__(self) -> None:
        # simple list: 1.33 GB of RAM -> 201799 terms
        #self._dictionary: list[Term] = [] #TODO Trie?

        # HashMap: 1.34 GB
        #self._dictionary: BooleanMap[str, Term] = HashMap() #TODO Trie?

        # Trie: 1.49 GB
        self._dictionary: BooleanMap[str, Term] = Trie()

    
    @classmethod
    def from_corpus(cls, corpus: dict[int, Document]) -> Self:
        # TODO build k-grams dictionary

        #temp_dict: dict[str, Term] = {}
        
        index = cls()
        for doc in corpus.values():
            if (doc.docID % 1000 == 0):
                print(f"Processing document with ID: {doc.docID}...")            
            tokens = tokenize(doc)
            for position, token in enumerate(tokens):
                term = Term(token, doc.docID, position)
                index._dictionary.add(token, term, Term.merge)
                # try:
                #     temp_dict[token].merge(term)
                # except KeyError:
                #     temp_dict[token] = term

        # index._dictionary = sorted(temp_dict.values())
        #index._dictionary = HashMap(temp_dict)
        #index._dictionary = Trie(temp_dict)
        return index
    
    def __getitem__(self, term: str) -> PostingList:
        # for t in self._dictionary:
        #     if (t.term == term):
        #         return t.posting_list
        # return PostingList()
        try:
            return self._dictionary[term].posting_list
        except KeyError:
            return PostingList()

class BooleanIRSystem:
    def __init__(self, corpus: dict[int, Document], index: InvertedIndex) -> None:
        self._corpus: dict[int, Document] = corpus
        self._index: InvertedIndex = index

    @classmethod
    def from_corpus(cls, corpus: dict[int, Document]) -> Self:
        index = InvertedIndex.from_corpus(corpus)
        return cls(corpus, index)
    
    def query(self, query: str) -> list[Document]:
        #TODO phrase query and wildcard
        words = query.split()
        normalized = (normalize(w) for w in words)
        posting_lists = (self._index[w] for w in normalized)
        plist = reduce(lambda x, y: x.intersection(y), posting_lists)
        return [self._corpus[docID] for docID in plist]
        
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
                doc = Document(movies_dict[plot[0]], plot[1])
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
        for r in result:
            print(r)
        print()
    else:
        exit = True

        


