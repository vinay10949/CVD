import requests
import json
import os
import pandas as pd

headers = {
    'accept': '*/*',
    'cache-control': 'no-cache',
    'content-type': 'application/json'
}

df=pd.read_csv(os.getcwd()+'/differential_tests/sample_payloads/test.csv') 
df.drop('cardio',axis=1,inplace=True)
df['json'] = df.apply(lambda x: x.to_dict(), axis=1) 
data=df['json'].tolist()
 
 
#data = '[{"active":0,"age":10950,"alco":1,"ap_hi":175,"ap_lo":80,"cholesterol":3,"gender":1,"gluc":3,"height":165,"smoke":0,"weight":120}]'

response = requests.post('http://localhost:5000/v1/predictions/predict', json=data,headers=headers)
print(response.status_code)
print(response.text)
print(response.reason)

