from json import load
import pandas as pd

def test_load():
    
    data_path = 'data/census.csv'
        
    df = pd.read_csv(data_path)
    
    assert type(df) == type(pd.DataFrame()), "TypeError: Type should be pandas Dataframe"