from utils import *
from scipy.sparse import csr_matrix
from collections import Counter
import math



class Invertedfile_TFIDF:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.processed_q2_list = data['processed_ques2'].tolist()

        print("Build vocabulary...")
        self.vocab = build_vocab(self.data['processed_ques2'])
        self.tfidf_cols = []
        for key, val in self.vocab.items():
            self.tfidf_cols.append(key) 

        print("Build TF-IDF matrix...")
        self.tfidf_matrix = self.transform()
        self.tfidf_array = self.tfidf_matrix.toarray()

        print("Build inverted file...")
        self.inv_idx_dict = self.generate_inverted_file()

        print("Model is ready to query!")

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
        sim = self.get_similarity_tfidf(query, k)
        self.print_top_k(sim)
        return 

    def accuracy(self, df: pd.DataFrame, k):
        """
        function to perform inverted file TFIDF in for each row and return the accuracy
        :params:
            df: Test dataframe
            k: number of similar documents
        """
        tmp = 0
        for i in df.index: # for each row
            query = df['question1'][df.index == i].values[0]
            ques_sim = self.get_similarity_tfidf(query,k)
            for q in ques_sim: #for each similar question
                if q[0] == i:
                    tmp += 1
                    break
        return tmp/len(df)

    def get_similarity_tfidf(self, query: str, k: int) -> dict:
        """"
        function to generate top similar document given query
        :params:
            query: query to find similar document
            k: number of similar document to return
        """
        global docs
        global vocab_idx
        doc_sim = {}
        query = text_preprocessing(query) # preprocess input sentence
        tokens = nltk.word_tokenize(query)
        for word in tokens: #for each word in query
            if word in self.inv_idx_dict.keys() : # if word in inverted index
                docs = self.inv_idx_dict[word]     # get questions having given word
                vocab_idx = self.vocab.get(word, -1)  # get word's position

            if vocab_idx != -1:            #  if word in the vocabulary
                for doc in docs:
                    doc_score = self.tfidf_array[doc][vocab_idx]   #get tfidf score
                    #accumulate the TFIDF score for each occured document
                    if doc in doc_sim.keys(): #alr record of doc
                        doc_sim[doc] += doc_score
                    else: #new doc
                        doc_sim[doc] = doc_score

        sorted_sim = sorted(doc_sim.items(), key = lambda item: item[1],reverse = True ) # sort by TFIDF score in descending order
        return sorted_sim[0:k] # return top k similar document

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

    def transform(self) -> csr_matrix:
        """
        function to generate TF-IDF matrix
        params:
            data: list of documents
            vocab: vocabulary
        """
        rows_idx = []              # stores which row number/question a word belongs to
        words_freq = []   # stores the frequency of a word in the vocabulary 
        results = []           # stores TF IDF value of a word
        
        for indx, row in enumerate(self.processed_q2_list): # for each row
            sentence_word_freq = dict(Counter(row.split())) # get the frequency of each word in the row
            for word, freq in sentence_word_freq.items(): # for each word in the row
                vocab_idx = self.vocab.get(word, -1)      # get the word index in the vocabulary
            
                if vocab_idx != -1:

                    rows_idx.append(indx)   
                    words_freq.append(vocab_idx)

                    doc_freq = self.get_doc_freq(word) # document frequency of a word
            
                    TF = freq / len(row.split())        # term frequency
                    IDF = math.log2(len(self.processed_q2_list)/doc_freq)     # inverse document frequency
                    
                    val = TF * IDF 
                    results.append(val)
        return csr_matrix((results, (rows_idx, words_freq)), shape=(len(self.processed_q2_list),len(self.vocab))) 

    def generate_inverted_file(self) -> dict:
        """
        function to generate inverted index from given corpus
        :params:
            data: list of documents
        """
        inv_idx_dict = {}

        for index, ques in enumerate(self.processed_q2_list):
            for word in nltk.word_tokenize(ques):
                if word not in inv_idx_dict.keys() : # new word
                    inv_idx_dict[word] = [index]     
                elif index not in inv_idx_dict[word] : # new document
                    inv_idx_dict[word].append(index) 

        return inv_idx_dict
