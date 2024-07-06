# BooleanSearchEngine
Welcome to BooleanSearchEngine! A powerful boolean Information Retrieval system
After indexing all documents provided, you can query the system through this console.

## How to run
This project needs Python 3.11 (or above) and `pip` installed

Clone this repository, run `pip install -r requirements.txt` inside the directory to install the dependencies (only one actually)

Run using `python3 main.py` or just `python main.py`

## How to use
Instructions:

This special symbols can be used to query the system:

Operators:

- `&` - to make AND queries
    
- `|` - to make OR queries

- `-` - to exclude a word (or a sub-query) from the results

Other symbols:

- `*` -  to do a wilcard query, with one or more wildcards per word!
- `" "` - to make a phrase query that matches exactly what is inside the quotes
- `( )` - to specify a sub-query (required if more than operator is used)

If you want to do a simple AND query between words, phrases, or words with wildcards
you can just write them separated by spaces without the need to use the AND operator and the parenthesis

Stemming is used to increase recall, but if you want to search the exact words just enclose them inside quotes

Search is case-insensitive

Examples of queries:

- `hunger games` -> equivalent to (hunger & games) it searches the words separately and intersects the results

- `(iron & ma*) -mask` -> searches the words "iron" and the words that start with "ma" but excludes "mask" from the results

- `(the & cat) -"the cat"` -> searches the words "the" and "cat" but not if are consecutives

- `(cat | dog)` -> searches for a myriad of results containing the words "cat" or "dog" or both

- `((cat -"cats") | (do* - "the dog"))` -> more and more queries can be nested!
