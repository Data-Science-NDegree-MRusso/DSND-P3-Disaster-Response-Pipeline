# Disaster Response Pipeline Project

## Overview

This Project is submitted as part of the Udacity Data Science Nanodegree.

For it the goal is to analyze disaster data provided by a company called Figure Eight (now part of [Appen](https://appen.com/)) that is partner in the Nanodegree, and to build a model for an API that classifies disaster messages.

In the [`data_files`](./data/data_files) folder, you'll find 2 csv files containing real messages that were sent during disaster events. The project includes a machine learning pipeline to categorize these events so that the messages could be sent to an appropriate disaster relief agency.

The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


## Requirements
In order to facilitate the execution of the Notebooks and of the scripts I have prepared an [`environment.yml`](./environment.yml) file to be used to install an environment with [Anaconda](https://www.continuum.io/downloads):

```sh
conda env create -f environment.yml
```

After the installation the environment should be visible via `conda info --envs`:

```sh
# conda environments:
#
dsnd-webdev        /usr/local/anaconda3/envs/dsnd-proj3
...

```


## Instructions
The code in this repo includes 2 jupyter notebooks (in the [`notebooks`](./notebooks) folder) and a three scripts. 

* The notebooks are provided as guidelines/references for the scripts. They do not need to be executed to run the webapp:
    - [`ETL Pipeline Preparation`](./notebooks/ETL_Pipeline_Preparation.ipynb) documents a step-by-step process to load data from the `.csv` files and save them in an SQL-lite DB;
    - [`ML_Pipeline_Preparation`](./notebooks/ML_Pipeline_Preparation.ipynb) documents a step-by-step process to load data from the DB generated previously and train a classifier on them.

* In order to use the scripts to set up the database and model, you'll need to execute the following commands in the project's root directory:
    - To run an ETL pipeline that cleans data and stores in database you'll need to run [`process_data.py`](./data/data_scripts/process_data.py):  
        `python data/data_scripts/process_data.py data/data_files/disaster_messages.csv data/data_files/disaster_categories.csv` _`{path to database file}`_;
    - To run a ML pipeline that trains a classifier, saves it in a pickle file and also saves a `.txt` file containing an evaluation report based on [`sklearn classification_report()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) you'll need to run [`train_classifier.py`](./models/models_scripts/train_classifier.py):  
        `python models/models_scripts/train_classifier.py` _`{path to database file}`_ _`{path to model file}`_ _`{path to report file}`_;
    - **Note** that the last argument for the previous script is optional: if you don't define a report file the outcome of the `classification_report()` will be displayed on screen.
        
 
* Finally, to run the webapp execute [`run.py`](./app/run.py) with the following command from the root directory:  
        `python app/run.py` _`{path to database file}`_ _`{path to model file}`_

* To see the webapp in your browser go to http://0.0.0.0:3001/ .
    
## Results
* A database containing the processed values is available in [`data/data_db`](./data/data_db) as `DisasterResponses.db`. The DB includes a single table called `DisasterResponses`.
* A pickle file containing a dictionary that includes a model and the datadets used to train/test it is available here
* An evaluation report for the model is available in [`models/models_files`](./models/models_files) 

All files where generated with the scripts above, and can be used to run the webapp.

## License
 <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
