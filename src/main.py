from src.Searcher import Searcher

if __name__ == "__main__":
    searcher = Searcher("sentiments_analysis.xlsx")
    searcher.tokenize()
    print('>>> tokenizing: ________________________________________________________________')
    print(searcher.data_frame[0])
    searcher.remove_stop_words()
    print('>>> remove stop words: __________________________________________________________')
    print(searcher.data_frame[0])
    searcher.stem_words()
    print('>>> stem words: _________________________________________________________________')
    print(searcher.data_frame[0])
    searcher.vectorize()
    print('>>> vectorize: __________________________________________________________________')
    print(type(searcher.matrix))
    searcher.tf_idf()
    print('>>> tf_idf: ______________________________________________________________________')
    print(searcher.matrix)

    searcher.to_csv("../out/matrix_file.csv")
    print(searcher.search_word("سلام"))
