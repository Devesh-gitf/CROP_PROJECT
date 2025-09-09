from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import uvicorn
import os

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="AI Crop Recommendation API", version="1.0")

# ---------------------------
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Serve frontend
# ---------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

# ---------------------------
# Input Models
# ---------------------------
class RecommendInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    humidity: float
    temperature: float
    rainfall: float

class CropRecommendation(BaseModel):
    crop: str
    score: float

class RecommendOutput(BaseModel):
    recommendations: List[CropRecommendation]

class FertilizerInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    target_crop: str

class FertilizerOutput(BaseModel):
    suggestions: Dict[str, str]

# ---------------------------
# Load & train model
# ---------------------------
df = pd.read_csv("crop_data.csv")

le = LabelEncoder()
df["crop"] = le.fit_transform(df["crop"])

X = df.drop("crop", axis=1)
y = df["crop"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# Endpoints
# ---------------------------
@app.post("/recommend", response_model=RecommendOutput)
def recommend(data: RecommendInput):
    input_data = pd.DataFrame([{
        "N": data.nitrogen,
        "P": data.phosphorus,
        "K": data.potassium,
        "ph": data.ph,
        "humidity": data.humidity,
        "temperature": data.temperature,
        "rainfall": data.rainfall
    }])
    probs = model.predict_proba(input_data)[0]
    crop_labels = le.inverse_transform(model.classes_)
    top_indices = probs.argsort()[-3:][::-1]
    recommendations = [
        CropRecommendation(crop=crop_labels[i], score=round(probs[i] * 100, 2))
        for i in top_indices
    ]
    return RecommendOutput(recommendations=recommendations)

@app.post("/fertilizer", response_model=FertilizerOutput)
def fertilizer(data: FertilizerInput):
    suggestions = {}
    suggestions["Nitrogen"] = (
        "Add Urea or Ammonium Sulfate" if data.nitrogen < 50 else "Nitrogen level is sufficient"
    )
    suggestions["Phosphorus"] = (
        "Add DAP or Rock Phosphate" if data.phosphorus < 40 else "Phosphorus level is sufficient"
    )
    suggestions["Potassium"] = (
        "Add MOP (Muriate of Potash)" if data.potassium < 40 else "Potassium level is sufficient"
    )
    suggestions["General"] = f"Ensure proper irrigation for {data.target_crop}"
    return FertilizerOutput(suggestions=suggestions)

# ---------------------------
# Run locally
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "ai_crop_recommender_backend:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )