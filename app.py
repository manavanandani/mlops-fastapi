from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")

# Define input format
class IrisRequest(BaseModel):
    features: list  # List of 4 numbers (e.g., [5.1, 3.5, 1.4, 0.2])

@app.post("/predict")
def predict(data: IrisRequest):
    input_array = np.array([data.features])
    prediction = model.predict(input_array)
    return {"prediction": int(prediction[0])}
