from typing import overload
import pandas as pd
from AverageWordEmbedding import AverageWordEmbedding
import numpy as np
import nltk
from utils import *
from collections import Counter
import math

class WeightedAverageEmbedding(AverageWordEmbedding):
    def __init__(self, file_path: str, dim: int, data: pd.DataFrame) -> None:
        self.data = data
        self.dim = dim
        self.file_path = file_path
        self.processed_q2_list = data['processed_ques2'].tolist()

        print("Build vocabulary...")
        self.vocab = build_vocab(self.data['processed_ques2'])

        print("Loading Glove Model...")
        self.glove_model = self.load_glove_model()

        print("Embedding All Question 2 data...")
        vec = []
        for ques in (self.data['processed_ques2'][0:5000]): 
            val = self.averaging(ques) 
            vec.append(val)
        self.q2_embedding = np.array(vec) # store average word embedding of all question 2

        print("Model is ready to query")

        


    #override
    def averaging(self, input) -> np.array:
        """
        function to calculate weighted average embedding
        """
        words_list = nltk.word_tokenize(input)
        tfidf_dct = self.tfidf_dict(input)
        tmp = [i for i in tfidf_dct.values()]
        tmp = np.array([[i/sum(tmp)] for i in tmp])
    
        vals = []

        for word in words_list:    
            if word in self.glove_model: # if word in glove model
                vals.append(self.glove_model[word])
        try:
            tmp2 = (np.array(vals) * tmp).sum(axis = 0)
        except:
            tmp2 = np.zeros(self.dim) # if no word in glove model, return zero vector
        return tmp2


    def get_doc_freq(self, word: str):
        """
        function to calculate document frequency:
        param:
            data: list of documents
            word: word to calculate document frequency
        """
        freq = 0
        for row in self.processed_q2_list:
            if word in row:
                freq = freq + 1
        return freq

    def tfidf_dict(self, sentence: str):
        """
        function to calculate tfidf for each word in vocabulary
        """
        tfidf_dict = {}
        sentence_word_freq = dict(Counter(sentence.split()))
        for word, freq in sentence_word_freq.items(): # for each word in sentence
            if word in self.vocab.keys(): # if word in vocabulary
                doc_freq = self.get_doc_freq(word) # document frequency of a word
                TF = freq / len(sentence.split())        # term frequency
                IDF = math.log2(len(self.processed_q2_list)/doc_freq)     # inverse document frequency
                val = TF * IDF
                tfidf_dict[word] = val
            else:
                tfidf_dict[word] = 0
        return tfidf_dict
