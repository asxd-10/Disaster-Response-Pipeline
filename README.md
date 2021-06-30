# Disaster-Response-Pipeline
Udacity ML Pipeline project for Disaster Response dataset by Figure-Eight

### Table of Contents

   + [Installation](#installation)
   + [Project Motivation](#project-motivation)
   + [File Descriptions](#file-descriptions)
   + [Conclusion and Inference](#conclusion-and-inference)
   + [Acknowledgement, Author and Licensing](#acknowledgement--author-and-licensing)

### Installation
The code should run with no issues using Python versions 3.* Using Jupyter notebook from Anaconda is recommended. You may use other data visualization tools like Tableau for reference. The libraries required are:-
* Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
* Natural Language Processing Libraries: NLTK
* SQL Database Libraries: SQLalchemy/SQLite

### Project Motivation
This is a project for the Udacity Nanodegree Data Scientist program.
The objective of the project is to get acquainted with ETL Pipeline, ML pipeline, NLP pipeline and Web Apps using Flask,Plotly. 

The overall motive is to design a Disaster Response Pipeline for actual datasets provided by [Figure-Eight](https://appen.com/)


### File Descriptions
[data](https://github.com/asxd-10/Disaster-Response-Pipeline/tree/master/data) - This data folder, attached to the repository contains all the data sets as well as the initial ETL pipeline.It contains different kinds of variables either numerical or categorical. The main database is [DisasterResponse.db](https://github.com/asxd-10/Disaster-Response-Pipeline/blob/master/data/DisasterResponse.db) - an SQL file , while the two csv files are [disaster_categories.csv](https://github.com/asxd-10/Disaster-Response-Pipeline/blob/master/data/disaster_categories.csv) and [disaster_messages.csv](https://github.com/asxd-10/Disaster-Response-Pipeline/blob/master/data/disaster_messages.csv)

### Conclusion and Inference
The important insights and findings obtained from the data analysis have been described in this [site](https://view6914b2f4-3001.udacity-student-workspaces.com/).

### Acknowledgement, Author and Licensing
For the project, I give credit to 
* [Figure-Eight](https://appen.com/) for the messages dataset
* [Udacity](https://classroom.udacity.com/) for providing the necessary guidance for this project

The licensing details of the dataset is available on [Udacity](https://classroom.udacity.com/). The code can be freely used by any individual or organization for their needs. MIT LICENSED.
Author - [Ashay Katre](https://github.com/asxd-10/)

Added a Webhook using Azure functions
