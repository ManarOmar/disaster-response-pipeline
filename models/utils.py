import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from sklearn.ensemble import AdaBoostClassifier



#define the regular expression for the URL
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

#define the English punctuations
punctuations_list = string.punctuation


def tokenize(text):
    """
    Tokenize function
    
    Arguments:
        text -> list of text messages (Arabic)
    Output:
        clean_tokens -> tokenized text, clean for ML modeling
    """
    
    #replacing any url with "urlplaceholder" string
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    #normalize text
    text = text.lower()
    
    #tokenizing the text message
    tokens = word_tokenize(text)
    
    #defining the lemmatization object 
    lemmatizer = WordNetLemmatizer()
    
    #defining a translator object to remove all punctations
    table = str.maketrans('', '', punctuations_list)

    clean_tokens = []
    #cleaning every token by stemming and removing punctations and appending to the clean list
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)     
        clean_tok = clean_tok.translate(table)        
        clean_tokens.append(clean_tok)
    
    #removing the stopwords from the clean_tokens list
    clean_tokens = [w for w in clean_tokens if w != '' and w not in stopwords.words('english')]


    return clean_tokens
