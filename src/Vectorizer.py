# import math
# import pandas as pd
# from collections import defaultdict
#
# class Vectorizer:
#     def __init__(self, data_frame):
#         self.data_frame = data_frame
#         self.num_docs = len(data_frame)
#         self.vectorizer = None
#
#
#     def fit(self):
#         # Create a set of all words in the corpus
#         corpus_words = set()
#         for text in self.data_frame:
#             for word in self.tokenize(text):
#                 corpus_words.add(word)
#
#         # Assign an index to each word in the corpus
#         self.word_to_id = {}
#         for i, word in enumerate(corpus_words):
#             self.word_to_id[word] = i
#
#         # Calculate the IDF values for each word in the corpus
#         self.idf = defaultdict(int)
#         for word in corpus_words:
#             # Count the number of documents that contain the word
#             count = sum(1 for text in self.data_frame if word in self.tokenize(text))
#             # Calculate IDF value for this word
#             self.idf[word] = math.log(self.num_docs / (1 + count))
#
#     def transform(self, text):
#         # Create a vector for the given text
#         words = self.tokenize(text)
#         vector = [0] * len(self.word_to_id)
#         for word in words:
#             if word in self.word_to_id:
#                 word_id = self.word_to_id[word]
#                 tf = words.count(word) / len(words)
#                 vector[word_id] = tf * self.idf[word]
#         return vector
#
#     def fit_transform(self):
#         # Fit the vectorizer and transform the documents to a TF-IDF matrix
#         self.fit()
#         self.matrix = []
#         for text in self.data_frame:
#             self.matrix.append(self.transform(text))
#         self.matrix = pd.DataFrame(self.matrix, columns=self.word_to_id.keys())
