import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

def load_data(messages_filepath, categories_filepath):
    """
    Load data frame from path file
    Args:
        messages_filepath: path to message file
        categories_filepath: path to category file
    Return:
        pandas.DataFrame: A merged data frame of the input files.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how ='outer', on =['id'])
    return df

def clean_data(df):
    """
    Clean and preprocess the input DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing unclean data.

    Returns:
        pandas.DataFrame: A cleaned and preprocessed DataFrame.
    """
     # Split the 'categories' column into separate columns
    categories = df['categories'].str.split(';', expand=True)
    categories.head()
    
    # Extract column names from the first row of the 'categories' DataFrame
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    
    # Assign meaningful column names to the 'categories' DataFrame
    categories.columns = category_colnames
    categories.head()
    
    # Convert category values to binary (0 or 1)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Drop the original 'categories' column from the input DataFrame
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the cleaned 'categories' DataFrame with the input DataFrame
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save a DataFrame to a SQLite database.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        database_filename (str): The filename of the SQLite database.

    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages', engine, index=False)  


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