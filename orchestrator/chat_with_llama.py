from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json
import requests
import re
import uuid 
from concurrent.futures import ThreadPoolExecutor 

app = Flask(__name__)
CORS(app)

# 用於儲存所有活躍對話的狀態。
conversation_store = {} 

# 單一使用者槽位的標準結構 (初始化模板)
SLOT_TEMPLATE = {
    "age": {"value": None},
    "sex": {"value": None},
    "smoker": {"value": None},
    "children": {"value": None},
    "region": {"value": None},
    "height": {"value": None},
    "weight": {"value": None},
    "bmi": {"value": None}
}

# ---------------------------
# Slot-Filling Prompt 
# ---------------------------
def build_slot_prompt(user_message, current_slots):
    known_data = {k: v["value"] for k, v in current_slots.items()}
    return f"""
你是一個保險資料抽取助手，請從使用者訊息中提取以下欄位：

age: number
sex: male/female
smoker: yes/no
children: number
region: string (如 台北市、高雄市)
height: number（公分）
weight: number（公斤）

⚠ 規則：
1. 只能輸出純 JSON，不能加文字。
2. 若使用者明確提供資訊，就填入。
3. 若未提到任何欄位，請保持為 null。
4. 不要自行詢問問題。

使用者訊息: "{user_message}"
目前資料: {json.dumps(known_data, ensure_ascii=False)}
"""

# ---------------------------
# 自動計算 BMI
# ---------------------------
def compute_bmi_if_possible(current_slots):
    h = current_slots["height"]["value"]
    w = current_slots["weight"]["value"]

    if h is None or w is None:
        return None

    try:
        h_m = float(h) / 100.0
        bmi = round(float(w) / (h_m ** 2), 2)
        current_slots["bmi"]["value"] = bmi 
        return bmi
    except:
        return None

# ---------------------------
# Chat Prompt 
# ---------------------------
def build_chat_prompt(user_message, current_slots):
    known_data = {k: v["value"] for k, v in current_slots.items()}
    missing = [k for k, v in known_data.items() if v is None and k not in ["bmi"]]

    chinese_keys = {
        "age": "年齡", "sex": "性別", "smoker": "是否吸菸",
        "children": "孩子數量", "region": "居住地",
        "height": "身高", "weight": "體重"
    }
    
    if missing:
        missing_chinese = [chinese_keys.get(k, k) for k in missing]
        missing_prompt_text = f"你目前還缺少這些重要資訊：{', '.join(missing_chinese)}。"
    else:
        missing_prompt_text = "目前所有欄位資訊已收集完畢。"

    return f"""
你是一位友善、親切且專業的保險規劃助理。

你已經知道使用者提供的資料是：
{json.dumps(known_data, ensure_ascii=False, indent=2)}

--- 任務要求 ---
1. **語氣：** 使用自然、親切、口語化的語氣回覆使用者。
2. **提問依據：** 你的提問必須基於以下你缺少的資訊：
    {missing_prompt_text}
3. **格式限制：** 你的回覆必須是 **一段連續的、純文本**。
4. **禁止符號：** **嚴禁** 使用任何 Markdown 格式符號來列出問題，例如：**星號 (\*)、列點符號 (-)、數字編號等**。請使用自然語句提問。
5. **不要重複** 已取得的資訊。

請根據使用者訊息和缺少的資訊，生成一段自然的回覆。
使用者訊息: "{user_message}"
"""

# ------------------------------------
# 最終諮詢 Prompt
# ------------------------------------
def build_final_consultation_prompt(price, products):
    """
    生成最終的完整回覆：包含價格宣告與產品推薦引導，並嚴格控制輸出格式。
    """

    return f"""
你是 **專業且友善的保險顧問 AI 助理**。

--- 任務與輸出要求 (請嚴格遵守) ---
1. **角色與語氣：** 專業、親切、自然、口語化，使用繁體中文。
2. **輸出格式：** 最終輸出 **只能** 是一段連續的、直接給客戶看的文字回覆。
3. **禁止輸出：** 嚴禁輸出任何分隔符、標題、列點符號、或任何形式的 **自我反思**、**解釋你的任務**、或 **重複 Prompt 內容**。

--- 內容要求 (必須包含以下資訊) ---
1. **開場/分析結果：** 感謝客戶提供的資訊，並告知系統已完成分析。
2. **預估價格宣告：** 簡潔明確地告知預估年保費約為 **{price} 元**。
3. **法律提醒：** 務必提醒客戶這只是基於模型的 **估計值**，**不具法律效力**，實際保費會根據最終核保結果和產品條款確定。
4. **產品引導：** 告知客戶系統已經推薦了最適合的三款相關產品，可以點擊下方連結查看詳情。

請根據上述要求，立即開始生成對客戶的完整文字回覆。
"""

# ---------------------------
# 呼叫 Ollama
# ---------------------------
def call_ollama(prompt_text):
    process = subprocess.Popen(
        ["ollama", "run", "llama3.1:8b"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = process.communicate(prompt_text)
    return out

# ---------------------------
# 抽出 JSON
# ---------------------------
def extract_json(text):
    try:
        # 嘗試直接解析
        if text.strip().startswith("{") and text.strip().endswith("}"):
            return json.loads(text)
        # 使用正則表達式尋找第一個和最後一個花括號之間的內容
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"JSON extraction failed: {e}")
        return None
    return None

# ---------------------------
# ML Predict 呼叫
# ---------------------------
def call_ml_predict(slots):
    try:
        res = requests.post(
            "http://localhost:5001/predict",
            json=slots,
            timeout=5
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"ML Predict Error: {e}")
        return {"predicted_charge": "無法計算"}

# ---------------------------
# Recommendation Service 呼叫
# ---------------------------
def call_recommendation_service(user_query):
    try:
        res = requests.post(
            "http://localhost:5003/recommend_products",
            json={"query": user_query, "top_k": 3},
            timeout=5
        )
        res.raise_for_status()
        if res.status_code == 200:
            return res.json().get("products", [])
        return []
    except Exception as e:
        print(f"Recommendation Error: {e}")
        return []

# ---------------------------
# Chat API
# ---------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # 獲取或初始化該對話的槽位狀態
    if conversation_id not in conversation_store:
        # 使用 .copy() 進行深拷貝，確保每個對話都有獨立的槽位字典
        conversation_store[conversation_id] = json.loads(json.dumps(SLOT_TEMPLATE)) 
    
    current_slots = conversation_store[conversation_id] # 取得當前對話的狀態

    # 1. Slot-Filling 
    slot_prompt = build_slot_prompt(user_message, current_slots)
    slot_output = call_ollama(slot_prompt)
    extracted = extract_json(slot_output)

    if extracted:
        # 使用 current_slots 進行更新
        for key in current_slots:
            if key in extracted and extracted[key] is not None and extracted[key] != "null":
                current_slots[key]["value"] = extracted[key]

    # 2. 自動計算 BMI
    compute_bmi_if_possible(current_slots)

    # 3. 檢查必填欄位
    required_fields = ["age", "sex", "smoker", "children", "region", "bmi"]
    complete = all(current_slots[k]["value"] is not None for k in required_fields)

    # ---------------------------------------------------------
    # 4. 資料收集完成後的流程
    # ---------------------------------------------------------
    if complete:
        slots_for_predict = {k: v["value"] for k, v in current_slots.items()}
        
        # 組合 RAG 查詢字串
        user_query_summary = f"客戶年齡 {slots_for_predict['age']}, 性別 {slots_for_predict['sex']}, BMI {slots_for_predict['bmi']}, {slots_for_predict['region']}人"
        if slots_for_predict['smoker'] == 'yes':
            user_query_summary += ", 有抽菸習慣"

        # A & B. 使用 ThreadPoolExecutor 進行並行呼叫 (ML Predict & RAG)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_price = executor.submit(call_ml_predict, slots_for_predict)
            future_recom = executor.submit(call_recommendation_service, user_query_summary)
            
            prediction = future_price.result()
            recommended_products = future_recom.result()

        transformed_products = []
        for p in recommended_products:
            transformed_products.append({
                **p, 
                "Summary": p.get("summary", p.get("Summary")), 
                "URL": p.get("url", p.get("URL"))              
                })
        
        charge = prediction.get("predicted_charge", "N/A")

        # C. 單次 Llama 生成完整回覆
        final_prompt = build_final_consultation_prompt(charge, transformed_products)
        final_consultant_reply = call_ollama(final_prompt).strip()

        # D. 回傳結果，務必包含 conversation_id
        return jsonify({
            "reply": final_consultant_reply,
            "slots": {k: v["value"] for k, v in current_slots.items()},
            "structured_data": {
                "predicted_price": charge,
                "recommendations": transformed_products
            },
            "complete": True,
            "conversation_id": conversation_id 
        })
        
    # 5. 資料尚未完成：繼續引導聊天
    chat_prompt = build_chat_prompt(user_message, current_slots) 
    chat_reply = call_ollama(chat_prompt).strip()

    return jsonify({
        "reply": chat_reply,
        "slots": {k: v["value"] for k, v in current_slots.items()},
        "complete": False,
        "conversation_id": conversation_id 
    })

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)