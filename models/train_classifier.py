import sys
import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier # ======================
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump #, load
import warnings
warnings.filterwarnings('always')

def load_data(database_filepath):
    '''load dataset form database table

    Args:
        database_filepath (str): database file name 
    
    Returns:
        X (Series): Pandas Series contains source "x" data for training/testing the model  
        Y (DataFrame): Pandas dataframe contains target "y" data for training/testing the model
        category_names (list): categories names 
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    conn = engine.connect()

    # read all the table to a dataframe
    df = pd.read_sql('Select * from categorizedMessages', con=conn)

    # specify X and Y values for training and testing the model
    X = df['message'].values[:1000]
    Y = df.iloc[:, 4:].values[:1000]

    category_names = df.iloc[:, 4:].columns

    return X, Y, category_names


def tokenize(text):
    '''tokenize, clean, and normalize text data for transformation 

    Args:
        text (str): untokenized, normalize, and uncleaned text data entry
    
    Returns:
        clean_tokens (list): cleaned, normalized, and tokenized text data in a list
    '''
    # find urls in the text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # replace all urls with a special token
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # case normalize, and lemmatize each token
    clean_tokens = []
    for token in tokens:
        token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(token)

    return clean_tokens



def build_model():
    '''instantiate model pipeline, parameters for grid search, and grid search object

    Args:
        None
    
    Returns:
        model (GridSearchCV): grid search object for training and infrence
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multilabel_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # parameters for grid search to choose the best
    parameters = {
        'tfidf__norm':['l1', 'l2'],
        'multilabel_clf__estimator__criterion': ['gini', 'entropy'],
        'multilabel_clf__estimator__n_estimators': [150, 200],
        'multilabel_clf__estimator__max_depth' : [4, 6]
    }

    # grid search object for the model and parameters
    model = GridSearchCV(estimator=pipeline, param_grid=parameters)
    
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    '''print classification report for model performance on each category in the testing data

    Args:
        model (GridSearchCV): trained estimator with best parameters
        X_test (numpy array): source data for testing
        Y_test (numpy array): target data for testing
        category_names (list): categories names 
    
    Returns:
        None
    '''
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        category = category_names[i]
        print('Category Name: ', category)
        print(classification_report(Y_test[:, i], Y_pred[:, i]))



def save_model(model, model_filepath):
    '''save trained model in given file path as pickle file

    Args:
        model (GridSearchCV): trained estimator with best parameters
        model_filepath (str): file path to save the model
    
    Returns:
        None
    '''
    dump(model, model_filepath) 


def main():
    '''main function runs all training, testing, and saving the model processes

    Args:
        None
        
    Returns:
        None 
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()