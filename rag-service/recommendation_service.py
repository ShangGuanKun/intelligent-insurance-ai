import os
from flask import Flask, request, jsonify
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
import torch

app = Flask(__name__)

# 使用環境變數或設定檔
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "insurance_products"

try:
    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Successfully connected to Chroma collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    collection = None

def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

model = load_model()

@app.route("/recommend_products", methods=["POST"])
def recommend_products():
    if collection is None:
        return jsonify({"error": "Database connection failed"}), 503

    try:
        data = request.get_json()
        query = data.get("query", "")
        top_k = data.get("top_k", 3)

        if not query:
            return jsonify({"error": "query is required"}), 400

        # Query embedding
        query_emb = model.encode(query).tolist()

        # 查詢 Chroma
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # 處理空結果的情況 
        if not results['ids']:
            return jsonify({"products": []})

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        products = []
        for i in range(len(ids)):
            meta = metas[i] or {}
        
            score = max(0.0, float(1 - distances[i])) 

            products.append({
                "id": ids[i],
                "score": round(score, 4),           # 取小數點後四位
                "title": meta.get("title", "未命名保險產品"), 
                "url": meta.get("url", "#"),        # 若無 URL 則給空連結
                "summary": docs[i]
            })

        return jsonify({"products": products})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)