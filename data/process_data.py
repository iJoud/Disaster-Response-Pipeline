import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''load dataset from given paths and combine it into a single dataframe

    Args:
        messages_filepath (str): dataset csv file path
        categories_filepath (str): dataset csv file path
    
    Returns:
        df (DataFrame): Pandas dataframe contains combined datasets
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge the two dataframes
    df = pd.merge(messages, categories, on ='id')

    return df

def clean_data(df):
    '''clean dataset dataframe and transform categories proper format 

    Args:
        df (DataFrame): dataset in Pandas dataframe 
    
    Returns:
        df (DataFrame): cleaned dataset in Pandas dataframe 
    '''
    categories = df['categories'].str.split(';', expand=True)

    # take row to clean its values and use as column names
    row = categories.columns = categories.iloc[0,:]
    category_colnames = [r[:-2].replace('_', ' ') for r in row]

    # rename columns with cleaned column names
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda val: val[-1:])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)


    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], sort=False, axis=1)
    
    # ensure all data are binary
    for col in df.iloc[:,4:].columns:
        df.drop(df[(df[col]> 1) | (df[col]< 0)].index, inplace=True)    

    # drop duplicated data, if exist!
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    '''save Pandas dataframe in database file in a table

    Args:
        df (DataFrame): dataset in Pandas dataframe 
        database_filename (str): database file name 
    
    Returns:
        None 
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('categorizedMessages', engine, index=False, if_exists='replace')  


def main():
    '''main function runs all Extract, Transform, and Load (ETL) processes

    Args:
        None
        
    Returns:
        None 
    '''
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
