# Disaster Response Pipeline Project
## Summary
In this project, I've build a web app shows brief Informations about training data and deploy multi-label classification model to classify disaster messages. The repo contains the ETL pipelines for preparing data, and training machine learning model.

## Files
data - Folder contains unprocessed data files and ETL pipeline scipt.
models - Folder contains machine learning pipeline.
app - Folder contains the html templates and flask python file to run the web app.


## Prerequisites & Running
You need to create an Anaconda environment with python 3.8 and the following libraries:
- sklearn
- json
- plotly
- pandas 
- numpy
- nltk 
- flask
- joblib
- sqlalchemy 
- re

After activating the environment, follow the following steps:
1. Run the ETL pipeline that cleans data and stores in database.
2. Run machine learning pipeline that trains classifier and saves it.
3. Finally, run web app and open localhost:3000 on your browser to open the web app. 



