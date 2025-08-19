from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("diabetes_model.pkl")

class Inputdata(BaseModel):
    Pregnancies : int
    Glucose : float
    BloodPressure : float
    BMI : float
    Age : int
    
@app.get("/")
def read_root():
    return {"message" : "Welcome to Nithi Diabetes Predictio API"}

@app.post("/predict")
def predict_diabetes(data: Inputdata):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]]);
    prediction = model.predict(input_data)[0]
    return {"prediction" : bool(prediction)}