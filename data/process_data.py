import sys
import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

def load_data(messages_filepath, categories_filepath):
  """
load the data from the csv files into dataframe

INPUT :
message_filepath -> csv file that handle the text messages
categories_file_path -> csv file that handle the categories for the messages ids

OUTPUT:
df -> the merged dataframefrom messages and categories
  """
    #read the two files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #merge the two dataframes on id
    df = pd.merge(messages, categories, on = 'id')

    return df

def clean_data(df):

  """
  cleaning the dataframe using these steps:
  1. create a dataframe of the 36 individual category columns
  2. extract a list of new column names for categories.
  3.rename the columns of categories and convert the type of them to string
  4.concatenate the original dataframe with the new categories dataframe
  5.drop duplicates

  INPUT -> uncleaned dataframe
  OUTPUT -> cleaned dataframe
  """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat = ";", expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    # extract a list of new column names for categories.
  
    category_colnames = row.apply(lambda x:x.split('-')[0]).values.tolist()
    
    # rename the columns of categories
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x.split('-')[1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)    
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
  """
  saving the cleaned dataframe to a database file
  INPUT:
  df -> the cleaned dataframe
  database_filename -> the name of the database file you want to save in
  """
    engine = create_engine('sqlite:///'+database)
    df.to_sql(sqltable, engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()