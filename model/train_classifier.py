import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from a SQLite database and prepare it for machine learning.

    Args:
        database_filepath (str): The filepath to the SQLite database.

    Returns:
        tuple: A tuple containing the following elements:
            - X (pandas.Series): The message data.
            - Y (pandas.DataFrame): The category labels.
            - category_names (Index): The names of category columns.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and lemmatize the input text.

    Args:
        text (str): The input text to be tokenized and lemmatized.

    Returns:
        list of str: A list of cleaned and lemmatized tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    """
    Build a machine learning model for multi-output classification.

    Returns:
        GridSearchCV: A GridSearchCV object for hyperparameter tuning.
    """
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('Tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    param_grid = {
        'classifier__estimator__n_estimators': [50, 100]
    }
    # Create a GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid)
    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a multi-output classification model and report F1 score, precision, and recall for each category.

    Args:
        pipeline: A scikit-learn pipeline containing the trained model.
        X_test: The test feature data.
        y_test: The true labels for the test data.

    Returns:
        None
    """
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    # Iterate through each output category and generate a classification report
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        report = classification_report(Y_test[category], y_pred[:, i])

def save_model(model, model_filepath):
    """
    Save a machine learning model to a file using pickle.

    Args:
        model: The trained machine learning model to be saved.
        model_filepath (str): The filepath where the model will be saved.

    Returns:
        None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

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