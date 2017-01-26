from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data_from_database(user, password, url, port, database, table):
    """
    Read arguments provided to pass to create_engine (using sqlalchemy) to return the 
    sql table in a pandas DataFrame.
    """
    
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(user, password, url, port, 
                                                                database))
    
    return pd.read_sql_table(table, con = engine)
            


def add_processes(process, data_dict):
    
    if 'processes' in data_dict:
        data_dict['processes'].append(process)
    else:
        data_dict['processes'] = [process]    
    
def make_data_dict(data, random_state = None, test_size = 0.25):
    """
    Performs a test train split, where 'X' features contain the string 'feat' and 'y' 
    target contains the string 'label' to return a dictionary of the train and test sets.
    """
    
    data_dict = {}
    
    X = data[[col for col in data.columns if 'feat' in col]]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state, 
                                                        test_size = test_size)
    
    data_dict['X'] = X
    data_dict['y'] = y
    data_dict['X train'] = X_train
    data_dict['y train'] = y_train
    data_dict['X test'] = X_test
    data_dict['y test'] = y_test
    
    return data_dict



def general_transformer(transformer, data_dict):
    """
    Returns a transformed data dictionary (which contains the X and y train and test 
    sets) when a transformer and data dictionary are passed through the function. Note,
    only the X train is fitted, and both the X train and test are transformed.
    """

    trans = transformer
    trans.fit(data_dict['X train'], data_dict['y train'])
    data_dict['X train'] = trans.transform(data_dict['X train'])
    data_dict['X test'] = trans.transform(data_dict['X test'])
    
    add_processes(transformer, data_dict)
    
    return data_dict



def general_model(model, data_dict):
    """
    Fit your train data and score your train and test set using the model of your choice.
    Be sure to pass your model through with parenthesis.
    """
    
    try:
        model.fit(data_dict['X train'], data_dict['y train'])
        
    except:
        raise ValueError('Make sure that you have split and transformed your data prior to modeling.')
        
    data_dict['train score'] = model.score(data_dict['X train'], data_dict['y train'])
    data_dict['test score'] = model.score(data_dict['X test'], data_dict['y test'])
    
    add_processes(model, data_dict)
    
    return data_dict