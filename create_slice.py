import pandas as pd
from starter.train_model import model_inference

def create_slice(datapath, slice_column, value = None):
    
    if value:
        df = pd.read_csv(datapath, index_col=None)
        df[slice_column] = df[slice_column].apply(lambda x: str(value))
    else:
        df = pd.read_csv(datapath, index_col=None)
        df[slice_column] = df[slice_column].apply(lambda x: df[slice_column][0])
    
    return df


if __name__ == '__main__':
    datapath = 'data/census.csv'
    slice_column = 'education'
    value = 'Bachelors'
    
    sliced_df = create_slice(datapath, slice_column, value)
    
    modelpath = 'model/randomforest.pkl'
    
    precision, recall, fbeta = model_inference(modelpath, sliced_df)
    
    with open('output_from_slice.txt','a') as f:
        result = f"Inference Performance \n\tcolumn: {slice_column}:{value}\n\tPrecision: {precision}\n\tRecall: {recall}\n\tF-beta score: {fbeta}\n\n"        
        f.write(result)