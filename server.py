from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

with open("models/xgboost_demand_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/api/categories")
def get_categories():
    categories = label_encoders["Category"].classes_.tolist()
    return {"categories": categories}


class PredictRequest(BaseModel):
    price: float
    discount: int
    inventory_level: int
    promotion: int
    competitor_pricing: float
    category: str


@app.post("/api/predict")
def predict(data: PredictRequest):
    input_data = pd.DataFrame({
        "Price": [data.price],
        "Discount": [data.discount],
        "Inventory Level": [data.inventory_level],
        "Promotion": [data.promotion],
        "Competitor Pricing": [data.competitor_pricing],
        "Category": [data.category],
    })

    for col, encoder in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = encoder.transform(input_data[col])

    prediction = model.predict(input_data)[0]
    return {"prediction": int(prediction)}
