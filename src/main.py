from src.Searcher import Searcher

if __name__ == "__main__":
    searcher = Searcher("english.xlsx")
    print('>>> tokenizing: ________________________________________________________________')
    searcher.tokenize()
    print(searcher.vocab)
    print('>>> remove stop words: _________________________________________________________')
    searcher.remove_stop_words()
    print(searcher.vocab)
    print('>>> stem words: _________________________________________________________________')
    searcher.stem_words()
    print(searcher.vocab)
    print('>>> vectorize: __________________________________________________________________')
    searcher.vectorize()
    print(searcher.vocab)
    print('>>> term document: ______________________________________________________________')
    searcher.calc_term_doc()
    print(searcher.term_doc_matrix)
    print('>>> tf_idf: _____________________________________________________________________')
    searcher.calc_tf_idf()
    print(searcher.term_doc_matrix)
    # to csv
    searcher.term_doc_matrix.to_csv('../out/term_doc_matrix.csv')

    print(searcher.search_query('movie drink'))
