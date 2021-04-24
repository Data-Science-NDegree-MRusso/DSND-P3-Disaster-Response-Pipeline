import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads data from csv files containing disater response messages and categories.

    First of all data are loaded in 2 independent DataFrame.
    Afterwards, categories are split and stored in separate coliumns, and converted
    from strings to numeric values.
    Finally, the datasets are combined and stored in a single pandas DataFrame,
    using a common id.

    Args:
        messages_filepath (str): file path for the messages file
        categories_filepath (str): file path for the categories file

    Returns:
        df: DataFrame containing combined messages/categories data
    """
    # Load messages/categories from files in 2 data sets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Separate `categories` id from values
    cat_id = categories.id
    cat_vals = categories.categories

    # Split cat_vals in 36 individual category columns
    cat_vals = cat_vals.str.split(';', expand=True)

    # Select the first row of the cat_vals dataframe, use this row to extract
    # a list of new column names for categories.
    row = cat_vals.loc[0]
    category_colnames = row.apply(lambda x: x[0:-2])

    # Rename the columns of `cat_vals` using the previous list
    cat_vals.columns = category_colnames

    # Iterate through the columns in cat_vals df to keep only the last character
    # of each string (the 1 or 0).
    for column in cat_vals:
        # set each value to be the last character of the string
        cat_vals[column] = cat_vals[column].str[-1]
        # convert column from string to numeric
        cat_vals[column] = pd.to_numeric(cat_vals[column])

    # Re-concatenate cat_id and cat_vals
    categories = pd.concat([cat_id, cat_vals], axis = 1)

    # Merge datasets
    df = messages.merge(categories,on=['id'])

    return df

def clean_data(df):
    """
    Cleans a dataset of disaster response messages and categories, removing all
    duplicates.

    Args:
        df: DataFrame containing combined messages/categories data

    Returns:
        df: Cleaned DataFrame containing combined messages/categories data without
            duplications
    """
    df = df.drop(df.index[df.duplicated()].tolist())

    return df


def save_data(df, database_filename):
    """
    Saves a dataset of disaster response messages and categories, in an SQL DB

    Args:
        df: Cleaned DataFrame containing combined messages/categories data
        database_filename (str): file path for the DB

    Returns:
        none
    """
    # For this we'll use sqlite
    engine_kind = 'sqlite:///'
    full_DB_engine = engine_kind + database_filename

    # Table name
    table_name = 'DisMesCat'

    # Create engine with above parameters and load data
    engine = create_engine(full_DB_engine)
    df.to_sql(table_name, engine, index=False)


def main():
    """
    Read two files with disaster response messages and categories, joins and
    processes them, and then load them in an sqlite DB.

    Takes as arguments the filepaths to the two files and the filepath to the DB
    """
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
