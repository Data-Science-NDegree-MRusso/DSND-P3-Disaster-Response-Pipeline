import json
import plotly
import pandas as pd
import re
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# NOTE: the following line was part if the original file but joblib has been
# deprecated in sklearn.externals starting from scikit-learn version 0.21.
# Instead it gets now imported directly
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

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

# Define flask app
app = Flask(__name__)

if len(sys.argv) == 3:
    # Read files paths
    database_filepath, model_filepath = sys.argv[1:]
else:
    # Use default values for file paths
    database_filepath = 'data/data_db/DisasterResponses.db'
    model_filepath = 'models/models_files/cv_trained_model.pkl'

# Load data
# For this we'll use sqlite
engine_dialect = 'sqlite:///'
# full_DB_engine_path = engine_dialect + app.config['db_path']
full_DB_engine_path = engine_dialect + database_filepath

# Assuming Table name
table_name = 'DisasterResponses'
# Create engine with above parameters and load data in a DataFrame
engine = create_engine(full_DB_engine_path)
df = pd.read_sql_table(table_name, engine)


# Load model
model_dict = joblib.load(model_filepath)
model = model_dict['model']

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract number of messages
    num_messages = df.shape[0]

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract/count the 10 most relevant categories
    cat_vals =  df[df==1][list(df.columns[4:])].sum().sort_values(ascending=False)[:10].values
    cat_names =  df[df==1][list(df.columns[4:])].sum().sort_values(ascending=False)[:10].index.values

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_vals
                )
            ],

            'layout': {
                'title': 'Top 10 Represented Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template(
        'master.html',
        num_messages=num_messages,
        ids=ids,
        graphJSON=graphJSON
    )

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

#Custom error route
#Documentation available at https://flask.palletsprojects.com/en/1.1.x/patterns/errorpages/
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


def main():
    if len(sys.argv) == 1:
        # Running with no arguments - Display message
        print('WARN: No arguments provided - looking for files in default locations')

    app.run(host='0.0.0.0', port=3001, debug=True)



if __name__ == '__main__':
    main()
