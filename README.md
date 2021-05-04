# Disaster Response Pipeline Project

## Overview

This Project is submitted as part of the Udacity Data Science Nanodegree.

For it the goal is to analyze disaster data provided by a company called Figure Eight (now part of [Appen](https://appen.com/)) that is partner in the Nanodegree, and to build a model for an API that classifies disaster messages.

In the [`data_files`](./data/data_files) folder, you'll find 2 csv files containing real messages that were sent during disaster events. The project includes a machine learning pipeline to categorize these events so that the messages could be sent to an appropriate disaster relief agency.

Your project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


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


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model, and to run the app:

    - To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/data_files/disaster_messages.csv data/data_files/disaster_categories.csv {path to database file}`
    - To run ML pipeline that trains classifier and saves a pickle file:
        `python models/train_classifier.py {path to database file} {path to model file}`
    - To run the web app: 
        `python app/run.py {path to database file} {path to model file}``

2. To see the webapp got to http://0.0.0.0:3001/


## License
 <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
