# import libraries
import nltk
nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords'])
import json
import joblib
import numpy as np
import pandas as pd
import string
import re
import sys
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sqlalchemy import create_engine
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils import tokenize
def load_data(database_filepath):
    """
    Load Data Function
    
    INPUT:
        no argument 
    Output:
        X -> feature(text message) array values
        Y -> label array values
        
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("messages", con = engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X,Y, category_names


def build_model():
    """
    building the model using machine learning pipelines 
    and grid search to validate the model
    """

    pipeline = Pipeline([
    ('tfidfvect',TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        'tfidfvect__ngram_range': ((1, 1), (1, 2)),
        'tfidfvect__max_df': (0.5, 1.0),
        'tfidfvect__max_features': (None, 5000),
        'clf__estimator__n_estimators': [50, 100] 
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate the model performance using accuracy, precision, f1 score
    Arguments:
        model -> the model which is built using the build_model function
        X_test -> the messages array in the testing dataset
        Y_test -> the labels array for the messages 
    Output:
        print the evaluation metrices  and the parameters for the best model specified by grid search 

    """

    y_pred = model.predict(X_test)

    for i, cat in enumerate(category_names):

        labels = np.unique(y_pred)
        confusion_mat = confusion_matrix(Y_test.iloc[:,i], y_pred[:,i], labels=labels)
        accuracy = (y_pred[:,i] == Y_test.iloc[:,i]).mean()
        class_report = classification_report(Y_test.iloc[:,i], y_pred[:,i])

        print("Labels:", labels)
        print("Confusion Matrix:\n", confusion_mat)
        print("Accuracy: ", accuracy)
        print("\nClassification report:\n ", class_report ) 

    print("\nBest Parameters: ", model.best_params_)

def save_model(model, model_filepath):
    """dumps the model to the given filepath
    Args:
        model -> the model which is built using the build_model function
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()