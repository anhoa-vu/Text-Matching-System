import pandas as pd
import numpy as np
import nltk
from utils import *

class AverageWordEmbedding:

    def __init__(self, file_path: str, dim: int, data: pd.DataFrame) -> None:
        self.data = data
        self.dim = dim
        self.file_path = file_path

        print("Loading Glove Model...")
        self.glove_model = self.load_glove_model()

        print("Embedding All Question 2 data...")
        vec = []
        for ques in (self.data['processed_ques2'][0:5000]): 
            val = self.averaging(ques) 
            vec.append(val)
        self.q2_embedding = np.array(vec) # store average word embedding of all question 2

        print("Model is ready to query")
    
    def query_engine(self) -> None:
        """
        function to return k most similar question given query
        :params:
            query: query to find similar document
            k: number of similar document to return
        """
        query = input("Enter query: ")
        k = int(input("Enter number of similar question to return: "))
        print(f"Your query is: {query}")
        sim = self.get_similarity_avg_word(query, k)
        self.print_top_k(sim)
        return 
    
    def accuracy(self, df: pd.DataFrame, k: int) -> float:
        """
        function to perform inverted file TFIDF in for each row and return the accuracy
        :params:
            df: Test dataframe
            k: number of similar documents
        """
        tmp = 0
        for i in df.index: # for each row
            query = df['question1'][df.index == i].values[0]
            ques_sim = self.get_similarity_avg_word(query,k)
            for q in ques_sim: #for each similar question
                if q[0] == i:
                    tmp += 1
                    break
        return tmp/len(df)

    def load_glove_model(self) -> dict:
        """
        function to load Glove model given file path
        """
        glove_model = {}
        with open(self.file_path,'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        return glove_model

    def neg_distance(self, pt1, pt2) -> float:
        """
        function to calculate negative euclidean distance between two points
        :params:
            pt1: first point
            pt2: second point
        """
        sum_sq = np.sum(np.square(pt1 - pt2))
        return -np.sqrt(sum_sq)

    def averaging(self, input) -> np.array:
        """
        function to vectorize the sentence using averaging word embedding method
        :params:
            input: input to be vectorized
        """
        words_list = nltk.word_tokenize(input)
        word_count = 0 
        
        for word in words_list:
            if word in self.glove_model:
                word_count += 1 
        vals = []

        for word in words_list:    
            if word in self.glove_model: # if word in glove model
                vals.append (self.glove_model[word])
        try:
            tmp = sum(vals)/word_count # sum of all word embedding and divide by number of word
        except:
            tmp = np.zeros(self.dim) # if no word in glove model, return zero vector
        return tmp

    def get_similarity_avg_word(self, query: str, k: int) -> dict:
        """"
        function to generate top similar document given query
        :params:
            query: query to find similar document
            k: number of similar document to return
        """
        distances = []
        doc_sim = {}
        query = text_preprocessing(query) # preprocess query text
        vec_query = self.averaging(query)

        tmp = 0
        for x in self.q2_embedding:
            try:
                sim_calculation = self.neg_distance(vec_query,x)
            except:
                sim_calculation = -99
            distances.append(sim_calculation)
            doc_sim[tmp] = sim_calculation 
            tmp += 1
        
        sorted_sim = sorted(doc_sim.items(), key = lambda item: item[1] ,reverse = True ) # sort by negative distance score in descending order
        return  sorted_sim[0:k]
    

    def print_top_k(self, k: list):
        """
        function to print top k similar document
        """
        for i in k:
            val = i[0]
            score = i[1]
            q = self.data['question2'][val]
            print(f" {q} - {score}")
            print("-----------------------------------------------------")
    