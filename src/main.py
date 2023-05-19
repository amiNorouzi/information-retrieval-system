from src.Searcher import Searcher

if __name__ == "__main__":
    searcher = Searcher("sentiments_analysis.xlsx")
    searcher.tokenize()
    searcher.remove_stop_words()
    # searcher.stem_words()
    # searcher.to_csv("../out/matrix_file.csv")
    # searcher.search_word("خوشحالی")
