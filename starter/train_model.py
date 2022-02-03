# Script to train machine learning model.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .ml.data import process_data
    from .ml.model import compute_model_metrics, inference, train_model
except:
    from ml.data import process_data
    from ml.model import compute_model_metrics, inference, train_model
import joblib
1

# Add the necessary imports for the starter code.

# Add code to load in the data.
def load_data(data_path):
    """
    Load the data from a source csv file
    
    Args:
        data_path (str): Path for the data csv file
    """
    
    df = pd.read_csv(data_path, index_col=None)
    
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    traindata, testdata = train_test_split(df, test_size=0.20, shuffle = True)
    
    return traindata, testdata


def training(traindata, modelpath, test=False):
    cat_features = [
        "workclass", 
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        traindata, categorical_features=cat_features, label="salary", training=True
    )
    # Train and save a model.
    model = train_model(X_train, y_train)
    
    if test == False:
        joblib.dump((model,encoder,lb), modelpath)
    else:
        return model
    
    pass

def model_inference(modelpath, testdata):
    cat_features = [
        "workclass", 
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    model, encoder, lb = joblib.load(modelpath)
    
    X_test, y_test, encoder, lb = process_data(
        testdata, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb = lb
    )
    
    y_pred = inference(model,X_test)
    
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    
    return precision, recall, fbeta


def main(datapath, modelpath):
    traindata, testdata = load_data(datapath)
    
    training(traindata,modelpath)
    
    precision, recall, f_beta = model_inference(modelpath, testdata)
    
    print(f'precision = {precision}, recall = {recall}, f_beta = {f_beta}')
    
    
if __name__ == '__main__':
    datapath = 'data/census.csv'
    modelpath = 'model/randomforest.pkl'
    main(datapath, modelpath)    