# Disaster Response Pipeline Project
## Summary
In this project, I've build a web app shows brief Informations about training data and deploy multi-label classification model to classify disaster messages. The repo contains the ETL pipelines for preparing data, and training machine learning model. It will help for fast disaster emergency response by categorize the income messages to appropriate categories.

## Files

```bash
    
    - app                           # Folder contains the html templates and flask python file to run the web app.
    ├── template                    
    │   ├── master.html             # main page of web app              
    │   └── go.html                 # classification result page of web app
    └── run.py                      # Flask file that runs web app
    
    - data                          # Folder contains unprocessed data files and ETL pipeline scipt.
    ├── disaster_categories.csv     # categories data to process 
    ├── disaster_messages.csv       # messages data to process
    ├── DisasterResponse.db         # created database file with cleaned dataset
    └── process_data.py             # ETL pipeline script for processing the data

    - models                        # Folder contains machine learning pipeline and trained model file.
    ├── train_classifier.py         # machine learning ETL script for the model
    └── classifier.pkl              # trained model saved
   
    - LICENSE
    - README.md
    
```



## Prerequisites & Running
You need to create an Anaconda environment with python 3.8+ and the following libraries:
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



