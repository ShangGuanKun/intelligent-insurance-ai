import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from pathlib import Path

# ------------------------------------------------------------
# 1. 讀取資料
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR.parent / "dataset" / "insurance.csv"
df = pd.read_csv(data_path)

# ------------------------------------------------------------
# 2. 資料前處理
# ------------------------------------------------------------

# sex → male:1 / female:0
df["sex"] = df["sex"].map({"male": 1, "female": 0})

# smoker → yes:1 / no:0
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

# region → 模擬成台灣地區
region_map = {
    "northeast": "Taipei",
    "northwest": "Taichung",
    "southeast": "Tainan",
    "southwest": "Kaohsiung",
}
df["region"] = df["region"].map(region_map)

# charges 換 TWD
df["charges"] = (df["charges"] * 30)

# ------------------------------------------------------------
# 3. Feature Engineering
# ------------------------------------------------------------
df["bmi_smoker"] = df["bmi"] * df["smoker"]
df["age_smoker"] = df["age"] * df["smoker"]
df["bmi_age"] = df["bmi"] * df["age"]

# ------------------------------------------------------------
# 4. 設定 X, y（log-transform）
# ------------------------------------------------------------
X = df[["age", "sex", "bmi", "children", "smoker",
        "region", "bmi_smoker", "age_smoker", "bmi_age"]]

y = np.log(df["charges"])   # log-transform

categorical_features = ["region"]
numeric_features = [col for col in X.columns if col != "region"]

# ------------------------------------------------------------
# 5. Preprocessor（OneHot + StandardScaler）
# ------------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)

# ------------------------------------------------------------
# 6. XGBoost
# ------------------------------------------------------------
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2,
        random_state=42
    ))
])

# ------------------------------------------------------------
# 7. 訓練 / 測試
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 8. 預測
# ------------------------------------------------------------
y_pred_log = model.predict(X_test)
y_pred = np.exp(y_pred_log)       # 還原成 TWD
y_true = np.exp(y_test)

# ------------------------------------------------------------
# 9. 評估
# ------------------------------------------------------------
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_test, y_pred_log)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)
#%%
# ------------------------------------------------------------
# 10. 儲存模型
# ------------------------------------------------------------

joblib.dump(model, "/Users/shangguankun/Downloads/Cathay/model/insurance_xgb_model.pkl")
print("模型已儲存 insurance_xgb_model.pkl")
