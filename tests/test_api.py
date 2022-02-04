from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_hello():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {"Hello": "Welcome!"}
    

def test_predict_income_1():
    data = {
        "age": 55,
        "workclass": "Local-gov",
        "fnlwgt": 130000,
        "education": "HS-grad",
        "education_num": 6,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    
    r = client.post('/inference', json=data)
    assert r.status_code == 200
    assert r.json() == {"income class": '<=50K'}, "Wrong Class"
    
    
def test_predict_income_2():
    data = {
        "age": 44,
        "workclass": "Private",
        "fnlwgt": 201723,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }
    
    r = client.post('/inference', json=data)
    assert r.status_code == 200
    assert r.json() == {"income class": '>50K'}, "Wrong Class"