# Disaster Response Pipeline Project

# Project Motivation
In this endeavor, I leveraged my data engineering expertise to conduct a comprehensive analysis of disaster-related data sourced from Figure Eight. The primary objective was to develop a robust machine learning pipeline capable of categorizing real-time messages transmitted during disaster scenarios. This classification system serves a critical role in directing these messages to the relevant disaster relief agencies. The culmination of this project is a user-friendly web application that empowers emergency responders to input new messages and swiftly obtain classification results across various categories. Additionally, the web app features insightful data visualizations, enhancing its utility for effective disaster response and management.

This project comprises three key components:

1. ETL Pipeline:
   - The `process_data.py` Python script constructs a robust data cleaning pipeline. It accomplishes the following tasks:
     - Loads both the messages and categories datasets.
     - Combines and merges the two datasets.
     - Conducts data cleaning and preprocessing.
     - Stores the cleaned data in a SQLite database.
   - Prior to creating the script, an exploratory data analysis (EDA) phase was undertaken using the Jupyter notebook "ETL Pipeline Preparation" to refine the `process_data.py` pipeline.

2. ML Pipeline:
   - The `train_classifier.py` Python script establishes a machine learning pipeline. Its main functionalities include:
     - Loading data from the SQLite database.
     - Partitioning the dataset into training and testing sets.
     - Constructing a text processing and machine learning pipeline.
     - Training and fine-tuning a model using GridSearchCV.
     - Producing evaluation results on the test set.
     - Exporting the final model as a pickle file.
   - A Jupyter notebook titled "ML Pipeline Preparation" was utilized during an EDA phase to facilitate the development of the `train_classifier.py` script.

3. Flask Web App:
   - The project incorporates a user-friendly web application that enables emergency personnel to input new messages and receive classification results across multiple categories.
   - Additionally, the web app provides insightful data visualizations to enhance data understanding and usability.
   - The application outputs are showcased below for user reference.

![image](https://github.com/quangnnb/udacity_data_scientist_project_2/assets/21145236/f2724392-fe5c-4d53-bb0d-67b2a1ca93e8)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
