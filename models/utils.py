import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import re
import string
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.ensemble import AdaBoostClassifier



#define the regular expression for the URL
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

#define the English punctuations
punctuations_list = string.punctuation


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

            return ' '.join(clean_tokens)
    
        return pd.Series(X).apply(tokenize).values


class UrlExtractor(BaseEstimator, TransformerMixin):
    """
    URL Extractor class
    
    This class extract the  'urlplaceholder' in the text,
    creating a new feature for the ML classifier
    """


    def finding_url(self, text):
        
        if 'urlplaceholder' in text:
            return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.finding_url)
        return pd.DataFrame(X_tagged)



class MessageLength(BaseEstimator, TransformerMixin):
    """
     Message Length Extractor class
    
    This class extract the message length ,
    creating a new feature for the ML classifier
    """

    def computing_message_length(self, text):
        return len(text)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.computing_message_length)
        return pd.DataFrame(X_tagged)


  