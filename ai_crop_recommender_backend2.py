from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = FastAPI()

# Load CSV
CSV_FILE = "crop_data.csv"
df = pd.read_csv(CSV_FILE)

# Detect crop column automatically
crop_col_candidates = [c for c in df.columns if 'crop' in c.lower()]
if not crop_col_candidates:
    raise ValueError("No crop column found in CSV")
TARGET_COL = crop_col_candidates[0]

# Features
FEATURES = [c for c in df.columns if c != TARGET_COL]

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != TARGET_COL]

feature_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    feature_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df[TARGET_COL] = target_encoder.fit_transform(df[TARGET_COL])

# Train model
X = df[FEATURES]
y = df[TARGET_COL]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Request models
class CropRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    moisture: float
    avg_temp: float
    rainfall: float
    water_available: str
    previous_crop: str

class FertilizerRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    target_crop: str

def encode_features(input_dict):
    df_input = pd.DataFrame([input_dict])
    for col, le in feature_encoders.items():
        if col in df_input:
            # Handle unseen labels
            val = df_input.at[0, col]
            if val in le.classes_:
                df_input[col] = le.transform([val])
            else:
                # If unseen, assign a new class
                le_classes = list(le.classes_) + [val]
                le.classes_ = np.array(le_classes)
                df_input[col] = le.transform([val])
    return df_input

@app.post("/recommend")
def recommend_crop(data: CropRequest):
    input_dict = data.dict()
    df_input = encode_features(input_dict)
    probs = model.predict_proba(df_input)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            "crop": target_encoder.inverse_transform([idx])[0],
            "score": round(probs[idx]*100, 2)
        })
    return {"recommendations": recommendations}

@app.post("/fertilizer")
def suggest_fertilizer(data: FertilizerRequest):
    target_crop_encoded = target_encoder.transform([data.target_crop])[0]
    crop_row = df[df[TARGET_COL] == target_crop_encoded].iloc[0]

    suggestions = {}
    for nutrient in ['nitrogen', 'phosphorus', 'potassium']:
        current_val = getattr(data, nutrient)
        ideal_val = crop_row[nutrient]
        diff = ideal_val - current_val
        if diff > 0:
            suggestions[nutrient] = f"Add {round(diff,2)} units"
        elif diff < 0:
            suggestions[nutrient] = f"Reduce {abs(round(diff,2))} units"
        else:
            suggestions[nutrient] = "Optimal"
    return {"suggestions": suggestions}