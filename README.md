# Salary Estimates
![alt text](http://www.blr.com/html_email/images/WIR/HRDA/HRDA_111516.jpg)

This project is an example of predicting job salary based on features such as job type, education, and 
other related features. The training and test data cannot be shared as the project was completed for a 
company using internal data. However, the code repo is provided as an example of conducting model training and price predictions.

## Getting Started

Project scripts
* requirements.txt file
* utils folder with constants.py and utils.py
* EDA.py
* clean_data.py
* train.py
  
Folders
* data folder. For model training and to run predictions it needs to contain train_features.csv, train_salaries.csv, and test_features.csv
* mlruns folder which tracks previous training results
* data which includes data used for training/test and model artifacts

## Installation

- Create a python virtual environment in the system and activate it.

**Installation using pip:**
- `pip install virtualenv`
- `virtualenv <env_name>`
- `source <env_name>/bin/activate`

Install the dependencies for the project using the requirements.txt
- `pip install -r requirements.txt`


### Creating data set
Run python salary_estimates/clean_data.py from CLI. The default destination of the data is
"data". 
Two optional parameters
- --cleaned_data_destination: Path where cleaned data will be saved
- --test_data_destination: Path where test data will be saved


### Training
Run python salary_estimates/train.py

The default source of training data is in data/cleaned_data.csv.
Optional parameters can be passed for model training including:
- --subset_data: The % of cleaned_data.csv to use before train/test split
- --val_size:  The % of data used for validation
- --eta: The algorithm's learning rate
- --max_depth: Max depth of a tree
- --n_estimators:  Number of trees
- --model_folder: Folder to store model artifacts
- --experiment_id: Experiment id

### Run predictions
Run python salary_estimates/predict.py
It's simplest to change the MODEL_FOLDER variable in the constants.py file once a new model has been trained.
Otherwise, the model_path argument can be passed when running the predict.py file which specified 
the folder for the model that will be used to run predictions.

- --model_folder: path to model folder used to make predictions
- --test_data_path: Path to test_features.csv file to run predictions on.
- --predictions_destination: Where to output predictions.

### Run test_cleaning_functions unit test
Run python tests/test_cleaning_functions.py
- --cleaning can be used to run tests for scope 'cleaning'
Note: 