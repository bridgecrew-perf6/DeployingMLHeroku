import pandas as pd
import pytest

def test_load():
    
    data_path = 'data/census.csv'
    
    if data_path is None:
        pytest.fail("Provide a valid path to the csv file")
        
    df = pd.read_csv(data_path)
    
    return df