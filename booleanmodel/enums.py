import re
from enum import Enum

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
    WORD = re.compile('\\w+')
    # WILDCARD_WORD = re.compile('[\\w\\*]*(?:(?:\\w\\*)|(?:\\*\\w))[\\w\\*]*')
    WILDCARD_WORD = re.compile(f'[\\w\\{QuerySpecialSymbols.WILDCARD.symbol}]*(?:(?:\\w\\{QuerySpecialSymbols.WILDCARD.symbol})|(?:\\{QuerySpecialSymbols.WILDCARD.symbol}\\w))[\\w\\{QuerySpecialSymbols.WILDCARD.symbol}]*')
    OPERATOR = re.compile(f'[{QuerySpecialSymbols.AND.symbol}{QuerySpecialSymbols.OR.symbol}{QuerySpecialSymbols.MINUS.symbol}]')
    OPEN_PARENTHESIS = re.compile(f'\\{QuerySpecialSymbols.OPEN_PARENTHESIS.symbol}')
    CLOSED_PARENTHESIS = re.compile(f'\\{QuerySpecialSymbols.CLOSED_PARENTHESIS.symbol}')
    QUOTE = re.compile(QuerySpecialSymbols.QUOTE.symbol)

    def matches(self, string: str) -> bool:
        m = re.fullmatch(self.value, string)
        return m is not None
