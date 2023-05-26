import re
from math import isinf

import nltk
import numpy as np
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords
from pandas import DataFrame

from src.Tokenizer import Tokenizer
from src.utils import to_path

nltk.download('stopwords')


class Searcher:
    # ? static properties
    english_stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    # ? Column name for document index in DataFrame passed
    col_name_doc_index = 'docID'
    # ? Column name containing text for all documents in DataFrame
    col_name_doc_text = 'tags'

    def __init__(self, src: str):
        # ? read dataset
        self.data_frame = pd.read_excel(to_path(src), header=None)
        # ? Create docId and tags columns
        self.data_frame.columns = [Searcher.col_name_doc_index, Searcher.col_name_doc_text]
        # ? Adds label D to docIDs
        self.data_frame.docID = pd.Series(["D" + str(ind) for ind in self.data_frame.docID])
        # ? Vocabulary of the documents
        self.vocab: list[str] = []
        # ? Term document matrix
        self.term_doc_matrix: DataFrame = DataFrame()

    def calc_term_doc(self) -> None:
        # این کد یک DataFrame خالی با ستون‌هایی که نام آنها در ستون Searcher.col_name_doc_index قرار دارد و ردیف‌هایی
        # که نام آنها در لیست vocab وجود دارد، ایجاد می‌کند. این DataFrame به شکل اولیه ماتریس اسناد و اصطلاحات را برای
        # الگوریتم جستجوی معکوس ایجاد می‌کند. همچنین، با بهینه‌سازی با استفاده از fillna(0)، تمام مقادیر در ماتریس به
        # صفر تنظیم می‌شوند.

        self.term_doc_matrix = pd.DataFrame(columns=self.data_frame[Searcher.col_name_doc_index],
                                            index=self.vocab).fillna(0)

        # این کد، ماتریس اسناد - اصطلاحات را برای الگوریتم جستجوی معکوس ایجاد می‌کند. برای هر یک از اصطلاحات موجود
        # در لیست vocab و هر یک از سند‌های موجود در داده‌ی ورودی، تعداد تکرار آن اصطلاح در آن سند را محاسبه می‌کند و
        # این تعداد را در ماتریس مورد نظر در سطر متناظر با آن اصطلاح و در ستون متناظر با آن سند قرار می‌دهد. با استفاده
        # از fillna(0) نیز، این کد تمامی محل‌هایی که در ماتریس خالی هستند را با صفر پر می‌کند.

        for word in self.term_doc_matrix.index:
            for doc in self.data_frame[Searcher.col_name_doc_index]:
                freq = \
                    self.data_frame[self.data_frame[Searcher.col_name_doc_index] == doc][
                        Searcher.col_name_doc_text].values[0].count(word)
                self.term_doc_matrix.loc[word, doc] = freq

    def calc_tf_idf(self) -> None:
        total_docx = len(self.data_frame.docID.values)
        self.term_doc_matrix['document_frequency'] = self.term_doc_matrix.sum(axis=1)
        self.term_doc_matrix['inverse_document_frequency'] = np.log2(
            total_docx / self.term_doc_matrix['document_frequency'])

        for word in self.term_doc_matrix.index:
            for doc in self.data_frame.docID.values:
                tf_idf = 0
                if not isinf(self.term_doc_matrix.loc[word]['inverse_document_frequency']):
                    tf_idf = np.log2(1 + self.term_doc_matrix.loc[word, doc]) * np.log2(
                        self.term_doc_matrix.loc[word]['inverse_document_frequency'])
                if word in self.term_doc_matrix.index and word in self.term_doc_matrix.columns:
                    self.term_doc_matrix.loc[word, 'tf_idf_' + doc] = tf_idf

    @staticmethod
    def query_processing(query: str) -> str:
        """
         این تابع حذف کاراکترهای غیر الفبایی، تبدیل کلمات به حروف کوچک و
         حذف کلمات از لیست کلمات را انجام میدهد.
        """
        query = re.sub(r'\W', ' ', query)
        query = query.strip().lower()
        query = " ".join([
            Searcher.stemmer.stem(word) for word in query.split()
            if word not in Searcher.english_stop_words
        ])

        return query

    def query_score(self, query) -> None:
        """
        این قطعه کد، با استفاده از روش tf-idf میزان اهمیت هر کلمه در جستجوی کاربر را برای مجموعه اسناد مورد نظر
        محاسبه می‌کند. در این روش، برای هر کلمه در جستجوی کاربر، ابتدا تعداد تکرار آن در متن اصلی (freq) محاسبه
        می‌شود. سپس با استفاده از مفهوم tf-idf، میزان مهمیت هر کلمه در متن را محاسبه می‌کند. ضریب tf در اینجا برابر
        با np.log2(1 + freq) است که این خروجی آن را برای مقادیر دیگر در مجموعه اسناد که جایی که کلمه مورد نظر در آن
        وجود دارد، برای محاسبه tf-idf کاربرد ندارد. به علاوه، inverse_document_frequency هر کلمه که در اندازه گیری
        اهمیت آن کمک می‌کند، با استفاده از شمارش تعداد اسنادی که هر کلمه وجود دارد، محاسبه می‌شود. سپس tf و idf کلمه
        با هم ضرب شده و در ستون جدول query_tf_idf ثبت شده و برای محاسبه امتیاز هر سند در جستجوی کاربر به کار می‌رود.
        """
        for word in np.unique(query.split()):
            freq = query.count(word)
            if word in self.term_doc_matrix.index:
                tf_idf = np.log2(1 + freq) * np.log2(self.term_doc_matrix.loc[word].inverse_document_frequency)
                self.term_doc_matrix.loc[word, "query_tf_idf"] = tf_idf
                self.term_doc_matrix['query_tf_idf'].fillna(0, inplace=True)

    def cosine_similarity(self, document_index, query_scores):
        # document_index: list.
        # ? List of document ids.
        #
        # query_scores: str.
        # ? Column name in DataFrame containing query term tf-idf scores.

        cosine_scores = {}

        query_scalar = np.sqrt(sum(self.term_doc_matrix[query_scores] ** 2))

        for doc in document_index:
            doc_scalar = np.sqrt(sum(self.term_doc_matrix[doc] ** 2))
            dot_prod = sum(self.term_doc_matrix[doc] * self.term_doc_matrix[query_scores])
            cosine = (dot_prod / (query_scalar * doc_scalar))

            cosine_scores[doc] = cosine

        return pd.Series(cosine_scores)

    def retrieve_indices(self) -> list[int]:
        cosines = self.cosine_similarity(self.data_frame.docID.values, 'query_tf_idf')
        self.data_frame = self.data_frame.set_index(Searcher.col_name_doc_index)
        self.data_frame['scores'] = cosines

        return list(self.data_frame.reset_index().sort_values('scores', ascending=False).index)

    def tokenize(self) -> None:
        self.data_frame.tags = self.data_frame.tags.str.replace(",", " ")
        self.data_frame.tags = self.data_frame.tags.str.replace(r'\W', ' ')
        self.data_frame.tags = self.data_frame.tags.str.strip().str.lower()
        text = " ".join(self.data_frame.tags.values)
        tokenizer = Tokenizer()
        self.vocab = tokenizer.tokenize(text)

    def remove_stop_words(self) -> None:
        self.vocab = [
            word for word in self.vocab
            if word not in Searcher.english_stop_words
        ]

    def stem_words(self) -> None:
        other_stem_words: list[str] = []

        for word in self.vocab:
            other_stem_words.append(Searcher.stemmer.stem(word))

        self.vocab = other_stem_words

    def vectorize(self) -> None:
        self.vocab = np.unique(self.vocab)

    def search_query(self, query: str) -> list[int]:
        query = self.query_processing(query)
        self.query_score(query)

        return self.retrieve_indices()
