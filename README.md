## Verivox interview task
This is a simple task as part of an interview process for a position of Data Scientist.In this task I have to analyse the given “Census” dataset
and build a predictive model that can predict whether a person makes an income of over $50K or not. Moreover, highlight some of the drivers of the model.

## Requirements

* python version 3.6 or above
* conda 4.6.14
All the others requirements and packages that are needed for this project is listed in requirements.txt file

## For testing on hidden data
To start with testing the model on hidden dataset in Verivox please follow the commands below.
After navigating to the task directory please run the following commands:
 ```
conda create --name <new_env> --file <requirements.txt>
conda activate <new_env>
python test_pipeline.py --input-csv-directory-path <your test data directory path> --input-csv-file-name <your test csv file name>
 ```
## Pipeline Architecture
The Machine Learning is a part of interview process therefore, I have decided to keep the pipeline simple. The initial analysis and design I have decided the following architecture for the pipeline.
You can see the Context level and container level architecture in the architecture folder as well as below.

<p align="center">
    <img width="250px" height="350px" src="/architecture/Context.png"/>
</p>

<p align="center">
    <img width="300px" height="450px" src="/architecture/Container.png"/>
</p>

## For deployment and testing on web interface
After navigating to the task directory please run the following commands:
 ```
conda create --name <new_env> --file <requirements.txt>
conda activate <new_env>
python app.py
```
Navigate to http://127.0.0.1:5000/ and fill the data of your choice after that press "Predict Income" button

## For Re-training

After navigating to the task directory please run the following commands:
 ```
conda create --name <new_env> --file <requirements.txt>
conda activate <new_env>
python main.py --data-file-path' <your data directory with file name>
```
Moreover,results will be saved in the same data directory.

## Expected results

* Model performance evluation graphs and Classification report.
* Model feature importance / Shape Values Analysis
* Analysis report that contains detailed data analysis, different model evaluations, main drivers of the model

## Report

For detailed report and discussion please find the Task_Report in ./Report/Task_Report

## Final Model

Pre-trained model which will be used for testing can be found in ./models


## Time need to done this task

* Data exploration and analysis - 10 Minutes
* Setup working environement and walking skeleton - 10 Minutes
* Development and deliverables
    * Architecture diagrams 20 minutes
    * Coding and model training - 20 Minutes
    * Readme - 10 minutes
    * Report writing - 30 Minutes
* Testing and bug fixing - 10 Minutes