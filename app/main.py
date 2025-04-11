from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import load_artifacts, preprocess_input, make_prediction

app = FastAPI()

# Load model and encoders once on startup
model, scaler, gender_encoder, smoking_encoder = load_artifacts()

class InputData(BaseModel):
    Age: float
    Gender: str
    Hypertension: int
    Heart_Disease: int
    Smoking_History: str
    BMI: float
    HbA1c_Level: float
    Blood_Glucose_Level: float

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    processed = preprocess_input(input_dict, scaler, gender_encoder, smoking_encoder)
    prediction = make_prediction(model, processed)
    
    try:
        proba = model.predict_proba(processed)[0][1]
    except:
        proba = "Not available"

    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return {
        "prediction": result,
        "confidence": proba,
        "raw_input": input_dict
    }
