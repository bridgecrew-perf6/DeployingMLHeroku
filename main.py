# Put the code for your API here.
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from starter.train_model import model_inference

app = FastAPI()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

class Data(BaseModel):
    age: int = Field(..., example=32)
    workclass: str = Field(..., example="Private")
    fnlwgt: int = Field(..., example=314234)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=12)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Sales")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=1000)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="United-States")

@app.get("/")
async def say_hello():
    return{"Hello": "Welcome!"}

@app.post('/inference')
async def predict(input: Data):
    data = jsonable_encoder(input)
    cols = {"age":"age",
            "workclass":"workclass",
            "fnlwgt":"fnlwgt",
            "education":"education",
            "education_num":"education-num",
            "marital_status":"marital-status",
            "occupation":"occupation",
            "relationship":"relationship",
            "race":"race",
            "sex":"sex",
            "capital_gain":"capital-gain",
            "capital_loss":"capital-loss",
            "hours_per_week":"hours-per-week",
            "native_country":"native-country"
            }
    data = dict((cols[key],value) for (key,value) in data.items())
    df = pd.DataFrame(data, index = [0])
    print(df)
    modelpath = 'model/randomforest.pkl'
    prediction = model_inference(modelpath, df)
    
    return{"income class": prediction}
