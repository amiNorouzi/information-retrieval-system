import math
import os

import nltk
import numpy as np
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
        self.tf_idf_matrix = None
        self.vocab = None
        self.term_freq = None
        self.document_term_matrix = None
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
        pd.DataFrame(np.array(self.tf_idf_matrix)).to_csv(path)

    def tokenize(self):
        tokenizer = Tokenizer()
        self.data_frame = self.data_frame.apply(tokenizer.tokenize)

    # todo
    def vectorize(self):
        # convert to lowercase and remove punctuation
        self.data_frame = self.data_frame.apply(lambda x: x.lower())
        self.data_frame = self.data_frame.str.replace('[^ws]', '')

        # calculate term frequency
        self.term_freq = []
        for text in self.data_frame:
            word_list = text.split(' ')
            freq_dict = {}
            for word in word_list:
                if word in freq_dict:
                    freq_dict[word] += 1
                else:
                    freq_dict[word] = 1
            self.term_freq.append(freq_dict)

        # create vocabulary
        temp_vocab = []
        for freq_dict in self.term_freq:
            for word in freq_dict.keys():
                if word not in temp_vocab:
                    temp_vocab.append(word)
        self.vocab = sorted(temp_vocab)

        # create document-term matrix
        self.document_term_matrix = np.zeros((len(self.term_freq), len(self.vocab)))
        for i, freq_dict in enumerate(self.term_freq):
            for j, word in enumerate(self.vocab):
                if word in freq_dict:
                    self.document_term_matrix[i, j] = freq_dict[word]

    def tf_idf(self):
        num_documents = len(self.document_term_matrix)
        num_terms = len(self.document_term_matrix[0])

        # Step 1: Calculate TF
        tf_matrix = [[0] * num_terms for _ in range(num_documents)]
        for i in range(num_documents):
            total_terms = sum(self.document_term_matrix[i])
            for j in range(num_terms):
                if total_terms:
                    tf_matrix[i][j] = self.document_term_matrix[i][j] / total_terms
                else:
                    tf_matrix[i][j] = 0

        # Step 2: Calculate IDF
        idf_vector = [0] * num_terms
        for j in range(num_terms):
            num_documents_with_term = sum(
                [1 for i in range(num_documents) if self.document_term_matrix[i][j] > 0]
            )
            if num_documents_with_term > 0:
                idf_vector[j] = math.log(num_documents / num_documents_with_term)

        # Step 3: Calculate TF-IDF and build the matrix containing the terms
        tfidf_matrix = [[] for _ in range(num_terms)]
        for j in range(num_terms):
            term = self.term_index_map[j]
            idf = idf_vector[j]
            for i in range(num_documents):
                tfidf = tf_matrix[i][j] * idf
                tfidf_matrix[j].append((term, tfidf))

        self.tf_idf_matrix = tfidf_matrix
        # self.tf_idf_matrix = pd.DataFrame(self.tf_idf_matrix, columns=self.vector.get_feature_names_out())

    # def search_query(self, query):
    #
    #     df = pd.read_csv('TagsDatabase.csv',header=None)
    #
    #     df.columns = ['docID','tags']
    #     df.docID = pd.Series(["D"+str(ind) for ind in df.docID])
    #
    #     df.tags = df.tags.str.replace(","," ")
    #     df.tags = df.tags.str.replace(r'\W',' ')
    #     df.tags = df.tags.str.strip().str.lower()
    #
    #     if not path.exists('term_doc_matrix.csv'):
    #
    #         print("Nopeeee")
    #         all_text = " ".join(df.tags.values)
    #         vocab = np.unique(word_tokenize(all_text))
    #         vocab = [word for word in vocab if word not in stopwords.words('english')]
    #
    #         similarity_index = term_document_matrix(df,vocab,'docID','tags')
    #         similarity_index = tf_idf_score(similarity_index, df.docID.values)
    #
    #         similarity_index.to_csv('term_doc_matrix.csv')
    #
    #     else:
    #
    #         similarity_index = pd.read_csv('term_doc_matrix.csv')
    #         similarity_index = similarity_index.set_index('Unnamed: 0')
    #
    #     query = query_processing(query)
    #     similarity_index = query_score(similarity_index,query)
    #
    #     cosines = cosine_similarity(similarity_index, df.docID.values, 'query_tf_idf')
    #     indices = retrieve_index(df, cosines, 'docID')
    #
    #     return json.dumps(list(indices))
