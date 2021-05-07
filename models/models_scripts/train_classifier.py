import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from matplotlib import pyplot as plt

import re
import pickle
import time
import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix

def load_data(database_filepath):
    """
    Loads a dataset of disaster response messages and categories, from an SQL DB

    Args:
        database_filepath (str): file path for the DB

    Returns:
        X: numpy array of features (messages)
        y: numpy array of categories
        category_names: list of the category labels
    """
    # For this we'll use sqlite
    engine_dialect = 'sqlite:///'
    full_DB_engine_path = engine_dialect + database_filepath

    # Full SQL query, assuming Table name
    table_name = 'DisasterResponses'
    full_query = ('SELECT * FROM ' + table_name)

    # Create engine with above parameters and load data in a DataFrame
    engine = create_engine(full_DB_engine_path)
    df = pd.read_sql(full_query, engine)

    # Get features and labes from DataFrame
    X = df.message.to_numpy()
    y = df[df.columns[4:]].to_numpy()

    # Get category names
    category_names = list(df.columns[4:])

    return X, y, category_names


def tokenize(text):
    """
    Cleans and tokenizes a text.

    An input string is converted to lower case and punctuation is removed.
    After that tokens are identified through the NLTK tokenizer.
    Finally a lemmatizer reduces the tokens that are  not stop words to their
    root form

    Args:
        text (str): the text to process

    Returns:
        tokens: the list of processed tokens
    """
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    tokens_raw = word_tokenize(text)

    # Lemmatize and remove stop words
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens_raw if
        (word not in stopwords.words('english'))]

    return tokens


def build_model():
    """
    Creates a model for NLP and text classification.

    The model is built as a GridSearchCV object, including:
    * A Pipeline composed by:
        * A vectorizer transformer
        * A TFIDF transformer
        * A Multi-output classifier, using a Random Forest algorythm
    * A set of parameters to run a Grid search optimization

    Args:
        none

    Returns:
        cv: the GridSearchCV object
    """
    #  Define Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    # Define parameters
    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        # 'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names, report_file_path=[]):
    """
    Evaluate a model after training

    First used the model to predict data and then shows metrics

    Args:
        model: Trained models
        X_test: Test feature dataset
        y_test: Test categories dataset
        category_names: list of the category labels
        report_file_path: [OPTIONAL] file path where to save a text file with
        scoring report. If not provide, the score will be just shown on screen.

    Returns:
        none
    """
    # Predict on test data
    y_pred = model.predict(X_test)

    # Iterate
    if (len(report_file_path) > 0):
        # If a filepath is provided store in a file
        with open(report_file_path, "w") as text_file:
            print('-----------------------------------------------------------------------------------------', file=text_file)
            print('Date and time when this report was generated: ', datetime.datetime.now(), file=text_file)
            for ind_1 in range(y_pred.shape[1]):
                print('-----------------------------------------------------------------------------------------', file=text_file)
                print('Label = ', category_names[ind_1], file=text_file)
                c_rep = classification_report(y_test[:,ind_1], y_pred[:,ind_1], output_dict=True, zero_division=0)
                kk = list(c_rep.keys())
                for ind_2 in range(len(c_rep) - 3):
                    print('Value = ', kk[ind_2], ': precision = ', "{:.2f}".format(c_rep[kk[ind_2]]['precision']),
                        '; recall = ', "{:.2f}".format(c_rep[kk[ind_2]]['recall']),
                        '; f1-s =', "{:.2f}".format(c_rep[kk[ind_2]]['f1-score']),
                        '; support =', c_rep[kk[ind_2]]['support'], file=text_file)
    else:
        # If a filepath is NOT provided display to screen
        for ind_1 in range(y_pred.shape[1]):
            print('-----------------------------------------------------------------------------------------')
            print('Label = ', category_names[ind_1])
            c_rep = classification_report(y_test[:,ind_1], y_pred[:,ind_1], output_dict=True, zero_division=0)
            kk = list(c_rep.keys())
            for ind_2 in range(len(c_rep) - 3):
                print('Value = ', kk[ind_2], ': precision = ', "{:.2f}".format(c_rep[kk[ind_2]]['precision']),
                    '; recall = ', "{:.2f}".format(c_rep[kk[ind_2]]['recall']),
                    '; f1-s =', "{:.2f}".format(c_rep[kk[ind_2]]['f1-score']),
                    '; support =', c_rep[kk[ind_2]]['support'])


def save_model(X_train, X_test, y_train, y_test, model, model_filepath):
    """
    saves model and data in a pickle file

    Args:
        X_train: train feature dataset
        X_test: Test feature dataset
        y_train: Train categories dataset
        y_test: Test categories dataset
        model: Trained models
        model_filepath: Path to the pickle file

    Returns:
        none
    """
    # Create dictionary for pickle file
    model_dict = {'X_train':X_train,
                    'y_train':y_train,
                    'X_test':X_test,
                    'y_test':y_test,
                    'model':model}

    # Save dictionary
    pickle.dump(model_dict, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) >= 3:
        database_filepath = sys.argv[1]
        model_filepath = sys.argv[2]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        t0= time.clock()

        print('Training model...')
        model.fit(X_train, Y_train)

        t1 = time.clock() - t0

        print('Done. Elapsed time: ', t1, ' s')

        if len(sys.argv) == 4:
            report_file_path = sys.argv[3]
            print('Evaluating model...\n    REPORT: {}'.format(report_file_path))
            evaluate_model(model, X_test, Y_test, category_names, report_file_path)
        else:
            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(X_train, Y_train, X_test, Y_test, model, model_filepath)

        print('Trained model saved!')

    else:
        print('\nPlease provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl'\
              '\n\nOptionally you can also provide the filepath of a .txt '
              'report for the model score. If you don\'t, the information will'
              ' be displayed on screen.')


if __name__ == '__main__':
    main()
