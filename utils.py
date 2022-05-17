import string
import nltk
from nltk.corpus import stopwords
import pandas as pd


def remove_stopwords(review):
    """
    function to remove stopword
    """
    stopwords_list=set(stopwords.words('english'))
    tokens = nltk.word_tokenize(review)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopwords_list]
    
    review = ' '.join(filtered_tokens)    
    return review

def remove_punctuation(review: str):
    """
    function to remove punctuation for
    """
    punc_list = [i for i in string.punctuation]
    tokens = nltk.word_tokenize(review)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in punc_list]

    review = ' '.join(filtered_tokens)
    return review

def to_lowercase(review: str):
    """
    transfer all word to lowecase
    """
    return review.lower()

def lemmatizer(review: str):
    """
    function to perform lemmatization
    """
    wnl = nltk.WordNetLemmatizer()
    review = ' '.join([wnl.lemmatize(word) for word in review.split()])
    return review

def text_preprocessing(text: str):
    """
    function to perform all preprocessing
    """
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatizer(text)
    return text

def build_vocab(rows: pd.Series) -> dict:
    """function to build vocab from given corpus
    params:
        rows: pd.Series
    """
    tmp = [] 
    for row in rows:
        for word in nltk.word_tokenize(row):
            if word not in tmp: #check if the word alr in the list
                tmp.append(word)    
    vocab = {j:i for i,j in enumerate(tmp)} # add to dict
    return vocab