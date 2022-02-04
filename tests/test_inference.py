from json import load
import numpy as np
import pandas as pd
from starter import train_model

def test_inference():
    datapath = 'data/census.csv'
    _ , testdata = train_model.load_data(datapath)
    
    modelpath = "model/randomforest.pkl"
    
    precision, recall, fbeta = train_model.model_inference(modelpath, testdata)
    assert precision >= 0, "ValueError: Precision should be greater than or equal to 0"
    assert precision <= 1, "ValueError: Precision should be less than or equal to 1"
    assert recall >= 0, "ValueError: Recall should be greater than or equal to 0"
    assert recall <= 1, "ValueError: Recall should be less than or equal to 1"