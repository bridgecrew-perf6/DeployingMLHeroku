import requests

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
    
response = requests.get(url='https://ravp90-udacity-project.herokuapp.com/')
#response = requests.post(
#    url='https://ravp90-udacity-project.herokuapp.com/inference',
#    json=data
#)

print(response.status_code)
print(response.json())