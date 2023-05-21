import numpy as np
import pandas as pd
from hazm import word_tokenize
from nltk.corpus import stopwords
from pandas import DataFrame

from src.utils import to_path


class Searcher:
    # ? static properties
    persian_stopwords_file = open(to_path("persian_stopwords.txt"), 'r', encoding='utf-8')
    persian_stop_words = {x[:-1] for x in persian_stopwords_file.readlines()}
    english_stop_words = set(stopwords.words('english'))
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

    def calc_term_doc(self):
        """Calculate frequency of term in the document.

        parameter:
            data: DataFrame.
            Frequency of word calculated against the data.

            vocab: list of strings.
            Vocabulary of the documents

            document_index: str.
            Column name for document index in DataFrame passed.

            text: str
            Column name containing text for all documents in DataFrame,

        returns:
            vocab_index: DataFrame.
            DataFrame containing term document matrix.
            """

        self.term_doc_matrix = pd.DataFrame(columns=self.data_frame[Searcher.col_name_doc_index],
                                            index=self.vocab).fillna(0)

        for word in self.term_doc_matrix.index:

            for doc in self.data_frame[Searcher.col_name_doc_index]:
                freq = \
                    self.data_frame[self.data_frame[Searcher.col_name_doc_index] == doc][
                        Searcher.col_name_doc_text].values[
                        0].count(word)
                self.term_doc_matrix.loc[word, doc] = freq

        return self.term_doc_matrix

    def tf_idf_score(self, inv_df='inverse_document_frequency'):
        """
        Calculate tf-idf score for vocabulary in documents

        parameter:
            vocab_index: DataFrame.
            Term document matrix.

            document_index: list or tuple.
            Series containing document ids.

            inv_df: str.
            Name of the column with calculated inverse document frequencies.

        returns:
            vocab_index: DataFrame.
            DataFrame containing term document matrix and document frequencies, inverse document frequencies and tf-idf scores
        """
        total_docx = len(self.data_frame.docID.values)
        self.term_doc_matrix['document_frequency'] = self.term_doc_matrix.sum(axis=1)
        self.term_doc_matrix['inverse_document_frequency'] = np.log2(
            total_docx / self.term_doc_matrix['document_frequency'])

        for word in self.term_doc_matrix.index:

            for doc in self.data_frame.docID.values:
                tf_idf = np.log2(1 + self.term_doc_matrix.loc[word, doc]) * np.log2(
                    self.term_doc_matrix.loc[word][inv_df])
                if word in self.term_doc_matrix.index and word in self.term_doc_matrix.columns:
                    self.term_doc_matrix.loc[word, 'tf_idf_' + doc] = tf_idf

        return self.term_doc_matrix

    @staticmethod
    def query_processing(query: str):
        """
         این تابع حذف کاراکترهای غیر الفبایی، تبدیل کلمات به حروف کوچک و
         حذف کلمات از لیست کلمات را انجام میدهد.
        """
        # query = re.sub(r'\W', ' ', query)
        query = query.strip().lower()
        query = " ".join([
            word for word in query.split()
            if word not in Searcher.english_stop_words and word not in Searcher.persian_stop_words
        ])

        return query

    def query_score(self, query):
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

    def retrieve_index(self, data, cosine_scores, document_index):
        """
        Retrieves indices for the corresponding document cosine scores

        parameters:
            data: DataFrame.
            DataFrame containing document ids and text.

            cosine_scores: Series.
            Series containing document cosine scores.

            document_index: str.
            Column name containing document ids in data.

        returns:
            data: DataFrame.
            Original DataFrame with cosine scores added as column.
        """

        data = data.set_index(document_index)
        data['scores'] = cosine_scores

        return data.reset_index().sort_values('scores', ascending=False).head(10).index

    def tokenize(self):
        self.data_frame.tags = self.data_frame.tags.str.replace(",", " ")
        self.data_frame.tags = self.data_frame.tags.str.replace(r'\W', ' ')
        self.data_frame.tags = self.data_frame.tags.str.strip().str.lower()
        # tokenizer = Tokenizer()
        self.vocab = np.unique(word_tokenize(" ".join(self.data_frame.tags.values)))

    def remove_stop_words(self):
        self.vocab = [
            word for word in self.vocab
            if word not in Searcher.english_stop_words and word not in Searcher.persian_stop_words
        ]

    def search_query(self, query: str) -> list[int]:
        query = self.query_processing(query)
        self.query_score(query)

        cosines = self.cosine_similarity(self.data_frame.docID.values, 'query_tf_idf')
        indices = self.retrieve_index(self.data_frame, cosines, 'docID')

        return list(indices)


if __name__ == "__main__":
    searcher = Searcher("english.xlsx")
    searcher.tokenize()
    searcher.remove_stop_words()
    searcher.calc_term_doc()
    searcher.tf_idf_score()
    searcher.term_doc_matrix.to_csv('../out/term_doc_matrix.csv')
    print(searcher.search_query('25 batman tom'))