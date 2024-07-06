from abc import abstractmethod
from  collections.abc import MutableMapping, Mapping, ItemsView, Iterator, Callable, Hashable
from typing import Self, TypeVar, Generic, Optional, cast

H = TypeVar("H", bound=Hashable)
S = TypeVar("S")
T = TypeVar("T")

class BooleanMap(MutableMapping[S, T]):
        @abstractmethod
        def __init__(self, mapping: Mapping[S,T] | None = None, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
            ...

        @abstractmethod
        def union(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
            ...
        
        @abstractmethod
        def intersection(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
            ...

        @abstractmethod
        def merge(self, other: Self, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> Self:
            ...

        @abstractmethod
        def add(self, key: S, value: T, on_same_element_callback: Callable[[T, T], T] = lambda x, y: x) -> None:
            ...
        
        @abstractmethod
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
    
    def items(self): # type: ignore # items() shuold return a ItemsView
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
