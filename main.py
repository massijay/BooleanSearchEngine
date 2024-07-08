from booleanmodel.boolean_model import BooleanIRSystem, Document
from booleanmodel.enums import QuerySpecialSymbols
import csv
import os
import os.path

SAVE_DATA_DIRECTORY = "ir_data"
SAVE_DATA_INDEX_FILE = "index.pickle"
SAVE_DATA_CORPUS_FILE = "corpus.pickle"

instructions = f'''
Welcome to BooleanSearchEngine! A powerful boolean Information Retrieval system
After indexing all documents provided, you can query the system through this console.

This special symbols can be used to query the system:
Operators:
    {QuerySpecialSymbols.AND.symbol}   - to make AND queries
    {QuerySpecialSymbols.OR.symbol}   - to make OR queries
    {QuerySpecialSymbols.MINUS.symbol}   - to exclude a word (or a sub-query) from the results
Other symbols:
    {QuerySpecialSymbols.WILDCARD.symbol}   - to do a wilcard query, with one or more wildcards per word!
    {QuerySpecialSymbols.QUOTE.symbol} {QuerySpecialSymbols.QUOTE.symbol} - to make a phrase query that matches exactly what is inside the quotes
    {QuerySpecialSymbols.OPEN_PARENTHESIS.symbol} {QuerySpecialSymbols.CLOSED_PARENTHESIS.symbol} - to specify a sub-query (required if more than operator is used)

If you want to do a simple AND query between words, phrases, or words with wildcards
you can just write them separated by spaces without the need to use the AND operator and the parenthesis

Stemming is used to increase recall, but if you want to search the exact words just enclose them inside quotes
Search is case-insensitive

Examples of queries:
    > hunger games -> equivalent to (hunger & games) it searches the words separately and intersects the results
    > (iron & ma*) -mask -> searches the words "iron" and the words that start with "ma" but excludes "mask" from the results
    > (the & cat) -"the cat" -> searches the words "the" and "cat" but not if are consecutives
    > (cat | dog) -> searches for a myriad of results containing the words "cat" or "dog" or both
    > ((cat -"cats") | (do* - "the dog")) -> more and more queries can be nested!

'''

# User-defined function to parse documents into a list of Document
# Document class can be extended to provide richer content
def parse_corpus() -> list[Document]:
    file_movie_names = "documents/movie.metadata.tsv"
    file_movie_plots = "documents/plot_summaries.txt"
    movies_dict = {}
    corpus: list[Document] = []

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
                doc = Document(movies_dict[plot[0]], plot[1]) # REMOVED PLOT FOR DEBUG
                # doc = Document(movies_dict[plot[0]], "")
                corpus.append(doc)
            except KeyError:
                pass
    return corpus

def load() -> BooleanIRSystem | None:
    print("Loading index...")
    if (not os.path.exists(SAVE_DATA_DIRECTORY)):
        return None
    if (not os.path.isdir(SAVE_DATA_DIRECTORY)):
        return None
    if (not os.path.exists(f'{SAVE_DATA_DIRECTORY}/{SAVE_DATA_INDEX_FILE}')):
        return None
    if (not os.path.exists(f'{SAVE_DATA_DIRECTORY}/{SAVE_DATA_CORPUS_FILE}')):
        return None
    ir = BooleanIRSystem.load_from_file(SAVE_DATA_DIRECTORY)
    ir.query("test")
    return ir

def save(ir : BooleanIRSystem):
    os.makedirs(SAVE_DATA_DIRECTORY, exist_ok=True)
    ir.save_to_file(SAVE_DATA_DIRECTORY)

def main():
    try:
        print(instructions)
        ir = load()
        if (ir == None):
            print("Saved index not found")
            print("Reading the documents...", end=" ")
            corpus = parse_corpus()
            print("Done")
            print("Indexing, it may take some time, grab a cup of coffee in the meanwhile if you want...\n")
            ir = BooleanIRSystem.from_corpus(corpus)
            print("Saving index...")
            save(ir)
        print("\nSystem ready!")
        exit = False
        while (not exit):
            print("Write a query, !! to see the instrucions or $$ to exit the program\n\n>", end=" ")
            q = input()
            if (q != "$$"):
                if (q == "!!"):
                    print(instructions)
                try:
                    result = ir.query(q)
                    print()
                    if (len(result) == 0):
                        print("No results found...", end="")
                    for doc in result:
                        print(doc.title)
                        desc = doc.get_content_as_string()[:100]
                        if (len(desc) > 0):
                            print(f'{desc}{"..." if len(desc) > 0 else ""}')
                        print()
                    print()
                except Exception as ex:
                    print(ex)
            else:
                exit = True
                print("Bye!")
    except KeyboardInterrupt:
        print("Exiting...")
main()
