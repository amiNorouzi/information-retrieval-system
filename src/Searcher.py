import os

import nltk
import pandas as pd
from nltk.corpus import stopwords
from parsivar import FindStems
from sklearn.feature_extraction.text import TfidfVectorizer

from src.Tokenizer import Tokenizer
from src.utils import to_path, root_dirname

nltk.download('stopwords')


class Searcher:
    # ? static properties
    stemmer = FindStems()
    persian_stopwords_file = open(to_path("persian_stopwords.txt"), 'r', encoding='utf-8')
    persian_stop_words = [x[:-1] for x in persian_stopwords_file.readlines()]
    english_stop_words = set(stopwords.words('english'))
    vector = TfidfVectorizer()

    def __init__(self, src: str):
        # ? read dataset
        self.matrix = None
        self.data_frame = pd.read_excel(to_path(src))
        self.data_frame = self.data_frame['text']

    # ? delete stop words
    def remove_stop_words(self):
        def rm_per_sw(words: list[str]):
            main_words: list[str] = []
            for word in words:
                if word not in Searcher.persian_stop_words and word not in Searcher.english_stop_words:
                    main_words.append(word)
            return main_words

        self.data_frame = self.data_frame.apply(rm_per_sw)

    # ? ریشه کلملت (استمینگ)
    def stem_words(self):
        def stem(words: list[str]):
            other_stem_words: list[str] = []
            for word in words:
                other_stem_words.append(Searcher.stemmer.convert_to_stem(word))
            return other_stem_words

        self.data_frame = self.data_frame.apply(stem)
        self.data_frame = self.data_frame.apply(lambda x: " ".join(x))

    def to_csv(self, path: str):
        self.matrix.to_csv(path, header=True, index=False)

    def tokenize(self):
        tokenizer = Tokenizer()
        self.data_frame = self.data_frame.apply(tokenizer.tokenize)

    # todo
    def vectorize(self):
        self.matrix = Searcher.vector.fit_transform(self.data_frame).todense()

    # todo
    def tf_idf(self):
        self.matrix = pd.DataFrame(self.matrix, columns=self.vector.get_feature_names_out())

        # تابعی برای جستجوی وزن یک کلمه در ماتریس

    @staticmethod
    def search_word(word):
        # استفاده از ابجکت my_stemmer کلاس FindStems
        word = Searcher.stemmer.convert_to_stem(word)

        try:
            # خواندن از فایل csv و استخراج مقدار داده مربوط به کلمه وارد شده
            df = pd.read_csv(os.path.join(root_dirname, 'out', 'matrix_file.csv'))

            # مقدار وزن کلمه مورد نظر محاسبه می‌شود
            x = df[word].sum()

            # نرمال کردن وزن کلمه
            first = df.values.min()
            last = df.values.max()
            result = (x - first) / (last - first)

            return result
        except:
            print("Not Found")
