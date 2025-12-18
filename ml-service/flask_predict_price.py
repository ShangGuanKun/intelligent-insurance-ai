from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "insurance_xgb_model.pkl"

model = joblib.load(MODEL_PATH)

app = Flask(__name__)

SEX_MAP = {"male": 1, "female": 0}
SMOKER_MAP = {"yes": 1, "no": 0}

REGION_MAP = {
    "Taipei": ["台北市","新北市","桃園市","基隆市","宜蘭縣","新竹縣","新竹市","苗栗縣"],
    "Taichung": ["台中市","彰化縣","南投縣","雲林縣","嘉義縣","嘉義市"],
    "Tainan": ["台南市","高雄市","屏東縣"]
}

REQUIRED_FEATURES = [
    "age", "sex", "bmi", "children", "smoker", "region"
]

def map_region(city: str) -> str:
    for region, cities in REGION_MAP.items():
        if city in cities:
            return region
    return "Kaohsiung"


def validate_input(data: dict):
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")


def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df["sex"] = df["sex"].map(SEX_MAP).fillna(0).astype(int)
    df["smoker"] = df["smoker"].map(SMOKER_MAP).fillna(0).astype(int)
    df["region"] = df["region"].apply(map_region)
    df["bmi_smoker"] = df["bmi"] * df["smoker"]
    df["age_smoker"] = df["age"] * df["smoker"]
    df["bmi_age"] = df["bmi"] * df["age"]

    return df


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        validate_input(data)

        df = preprocess(data)
        y_pred_log = model.predict(df)
        y_pred = np.exp(y_pred_log)

        return jsonify({
            "predicted_charge": float(np.round(y_pred[0], 0))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
