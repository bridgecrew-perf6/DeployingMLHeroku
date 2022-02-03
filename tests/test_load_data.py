from json import load
import pandas as pd
import pytest
from starter.train_model import load_data, model_inference, training
from starter.ml.model import train_model, compute_model_metrics,inference
from sklearn.ensemble import RandomForestClassifier

def test_load():
    
    data_path = 'data/census.csv'
        
    df = pd.read_csv(data_path)
    
    assert(type(df) == type(pd.DataFrame), "TypeError: Type should be pandas Dataframe")
    

def test_model_type():
    datapath = 'data/census.csv'
    traindata, _ = load_data(datapath)
    
    modelpath = ''
    
    model = training(traindata,modelpath, test=True)
    rf = RandomForestClassifier()
    
    assert(type(model) == type(rf), "TypeError: Expected Random Forest classifier")
    
def test_inference():
    datapath = 'data/census.csv'
    _ , testdata = load_data(datapath)
    
    modelpath = "../models/randomforest.pkl"
    
    precision, recall, fbeta = model_inference(modelpath, testdata)
    
    assert(type(precision) == type(float), "TypeError: Precision should be a float")
    assert(type(recall) == type(float), "TypeError: Recall should be a float")
    assert(type(fbeta) == type(float), "TypeError: F_beta should be a float")
    assert(precision >= 0, "ValueError: Precision should be greater than or equal to 0")
    assert(precision <= 1, "ValueError: Precision should be less than or equal to 1")
    assert(recall >= 0, "ValueError: Recall should be greater than or equal to 0")
    assert(recall <= 1, "ValueError: Recall should be less than or equal to 1")