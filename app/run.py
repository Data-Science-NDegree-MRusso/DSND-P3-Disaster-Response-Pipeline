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

# nltk.download('stopwords');

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
    model = model_dict['pipeline']

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

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


def main():
    if len(sys.argv) == 3:
        app.run(host='0.0.0.0', port=3001, debug=True)
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \nExample: python '\
              'run.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
