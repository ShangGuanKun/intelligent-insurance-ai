import requests

# -------------------------------
# 1. API URL
# -------------------------------
url = "http://127.0.0.1:5001/predict"

# -------------------------------
# 2. 測試 JSON 資料
# -------------------------------
test_data = [
    {
        "age": 35,
        "sex": "male",
        "bmi": 25.0,
        "children": 1,
        "smoker": "no",
        "region": "台北市"
    },
    {
        "age": 50,
        "sex": "female",
        "bmi": 30.5,
        "children": 2,
        "smoker": "yes",
        "region": "高雄市"
    },
    {
        "age": 28,
        "sex": "female",
        "bmi": 22.0,
        "children": 0,
        "smoker": "no",
        "region": "台中市"
    }
]

# -------------------------------
# 3. 呼叫 API
# -------------------------------
for i, data in enumerate(test_data):
    try:
        # Add a timeout to prevent the request from getting stuck
        response = requests.post(url, json=data, timeout=5)
        if response.status_code != 200:
            print("HTTP Error:", response.status_code)
            continue
        result = response.json()
        print(f"Test case {i+1}:")
        print("Input:", data)
        print("Output:", result)
        print("-" * 50)
    except Exception as e:
        print(f"Test case {i+1} failed:", e)
