from json import load
import numpy as np
import pandas as pd
from starter import train_model
from sklearn.ensemble import RandomForestClassifier

def test_model_type():
    datapath = 'data/census.csv'
    traindata, _ = train_model.load_data(datapath)
    
    modelpath = ''
    
    model = train_model.training(traindata,modelpath, test=True)
    rf = RandomForestClassifier()
    
    assert type(model) == type(rf), "TypeError: Expected Random Forest classifier"