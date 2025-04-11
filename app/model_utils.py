import joblib
import numpy as np

def load_artifacts():
    model = joblib.load("app/artifacts/model.pkl")
    scaler = joblib.load("app/artifacts/scaler.pkl")
    gender_encoder = joblib.load("app/artifacts/gender_encoder.pkl")
    smoking_encoder = joblib.load("app/artifacts/smoking_encoder.pkl")
    return model, scaler, gender_encoder, smoking_encoder

def preprocess_input(data, scaler, gender_encoder, smoking_encoder):
    gender = gender_encoder.transform([data["Gender"]])[0]
    smoking = smoking_encoder.transform([data["Smoking_History"]])[0]

    features = [
        gender,
        data["Age"],
        data["Hypertension"],
        data["Heart_Disease"],
        smoking,
        data["BMI"],
        data["HbA1c_Level"],
        data["Blood_Glucose_Level"]
    ]

    scaled = scaler.transform([features])
    return scaled

def make_prediction(model, processed_data):
    return model.predict(processed_data)
