from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import joblib
svm_model = joblib.load('svm_model.pkl')
iso_forest = joblib.load('iso_forest.pkl')
scaler = joblib.load('scaler.pkl')
app = FastAPI()
class UserInputs(BaseModel):
    ph : float
    Hardness : float
    Solids : float
    Chloramines : float
    Sulfate : float
    Conductivity : float
    Organic_carbon : float
    Trihalomethanes : float
    Turbidity : float

@app.get('/')
def check_status():
    return{
        'status' : 'ok',
        'message' : 'Use predict endpoint to make Predictions'
    }

@app.post('/predict')
def predict_inpust(data:UserInputs):
    input_df = pd.DataFrame([data.model_dump()])
    scaled_df = scaler.transform(input_df)
    anomaly = int(iso_forest.predict(scaled_df)[0])
    prediction = int(svm_model.predict(scaled_df)[0])
    return{
        'is_anomaly' : anomaly,
        'Potability' : prediction
    }
