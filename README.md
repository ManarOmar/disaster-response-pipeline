# Disaster Response Pipeline Project

This project is showing some of my data engineering skills, from my believe of the importance of the communication between data scientists and data engineers about what data we need and what format the data needs to be in, that as data scientists we can't build machine learning models and extract features with messy poorly formatted data, that's where data engineering comes into play.

In this project I applied these skills to analyze disaster data from [figure eight](https://www.figure-eight.com) to build a model for an API that classifies disaster messages. I used a data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency using the API, that my project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app displays visualizations of the data.

## Files
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py
|- Disaster.db   # database to save clean data to

- models
|- train_classifier.py
|- utils.py
|- clf.pkl  # saved model


- ETL Pipeline Preparation.ipynb # etl exploration
- ML Pipeline Preparation.ipynb # ML exploration
- README.md
```
In this project, I  build, evaluate, and save the data model to achieve the best data science solution for this problem,I embedded the data science solution into the web app production.

### Requirements
* flask
* joblib
* pandas
* plot.ly
* numpy
* scikit-learn
* sqlalchemy


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster.db models/clf.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

