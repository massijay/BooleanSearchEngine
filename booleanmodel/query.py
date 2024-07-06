from booleanmodel.utilities import TextUtilities
from booleanmodel.enums import QuerySpecialSymbols, TokenType, OperatorType
from booleanmodel.entities import Index, PostingList
from booleanmodel.exceptions import InvalidQueryException, OperatorNotSupportedException
from functools import reduce
import abc
import re

class QueryNode(abc.ABC):
    @abc.abstractmethod
    def query(self, index: Index) -> set[int]:
        pass

class Token(QueryNode):
    def __init__(self, string: str) -> None:
        self._string = string

    def query(self, index: Index) -> set[int]:
        docIDs = index[self._string].get_docIDs()
        lem_str = TextUtilities.lemmatize(self._string)
        if (lem_str != self._string):
            docIDs.update(index[lem_str].get_docIDs())
        return docIDs
    
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
                if (p < 0):
                    # if bad position, skip (e.g lemmatized token with position -1)
                    continue
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
                raise OperatorNotSupportedException(f'Operator with symbol {self._type.symbol} is not supported')
            
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
        x = re.search("\\w+", string)
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
        query_string = TextUtilities.normalize(query_string)
        if (not cls._are_quotes_balanced(query_string)):
            raise InvalidQueryException("Unbalanced number of quotes")
        if (not cls._are_parenthesis_balanced(query_string)):
            raise InvalidQueryException("Unbalanced number of parenthesis")
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
                raise InvalidQueryException(f'Symbol "{ql[end]}" found before phrase end, expected quote symbol \'{QuerySpecialSymbols.QUOTE.symbol}\'')
            node = Phrase(ql[start:end])
            self._i = end + 1
            return node
        
        if (not TokenType.OPEN_PARENTHESIS.matches(ql[self._i])):
            raise InvalidQueryException(f'Word, quote symbol \'{QuerySpecialSymbols.QUOTE.symbol}\' or open parenthesis "{QuerySpecialSymbols.OPEN_PARENTHESIS.symbol}" expected, got "{ql[self._i]}"')
        self._i += 1
        left = self._parse()

        if (TokenType.CLOSED_PARENTHESIS.matches(ql[self._i])):
            self._i += 1
            return left

        if (not TokenType.OPERATOR.matches(ql[self._i])):
            raise InvalidQueryException(f'Valid Operator symbol or closed parenthesis "{QuerySpecialSymbols.CLOSED_PARENTHESIS.symbol}" expected, got "{ql[self._i]}"')
        op_symbol = ql[self._i]
        self._i += 1
        right = self._parse()

        if (not TokenType.CLOSED_PARENTHESIS.matches(ql[self._i])):
            raise InvalidQueryException(f'Closed parenthesis symbol "{QuerySpecialSymbols.CLOSED_PARENTHESIS.symbol}" expected, got "{ql[self._i]}"')
        self._i += 1
        return Operator(op_symbol, left, right)

    def execute(self, index: Index) -> set[int]:
        return self._root.query(index)
