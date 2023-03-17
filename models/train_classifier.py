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

    engine = create_engine(f'sqlite:///{database_filepath}')
    conn = engine.connect()

    df = pd.read_sql('Select * from categorizedMessages', con=conn)
    X = df['message'].values[:1000]
    Y = df.iloc[:, 4:].values[:1000]
    category_names = df.iloc[:, 4:].columns

    return X, Y, category_names


def tokenize(text):
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

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multilabel_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # 
    parameters = {
        'tfidf__norm':['l1', 'l2'],
        'multilabel_clf__estimator__criterion': ['gini', 'entropy'],
        'multilabel_clf__estimator__n_estimators': [150, 200],
        'multilabel_clf__estimator__max_depth' : [4, 6]
    }

    #
    model = GridSearchCV(estimator=pipeline, param_grid=parameters)
    
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        category = category_names[i]
        print('Category Name: ', category)
        print(classification_report(Y_test[:, i], Y_pred[:, i]))



def save_model(model, model_filepath):

    dump(model, model_filepath) 


def main():
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