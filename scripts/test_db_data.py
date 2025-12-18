import json
import time
import torch
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer


# ---------------------------
# 1. Chroma 連線
# ---------------------------
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "insurance_products"

client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(COLLECTION_NAME)


# ---------------------------
# 2. 顯示 Collections 資訊
# ---------------------------
def show_collections():
    print("=== 所有 Collections ===")
    for c in client.list_collections():
        print("-", c.name)


def show_collection_info():
    print("\n=== Collection 資訊 ===")
    print("Name:", collection.name)
    print("Count:", collection.count())


# ---------------------------
# 3. 顯示前 N 筆資料
# ---------------------------
def preview_documents(limit=5):
    print(f"\n=== 前 {limit} 筆資料內容 ===")
    results = collection.get(
        limit=limit,
        include=["documents", "metadatas"]
    )

    for i in range(len(results["ids"])):
        print(f"\n--- 第 {i+1} 筆 ---")
        print("ID:", results["ids"][i])
        print("Metadata:", json.dumps(results["metadatas"][i], ensure_ascii=False, indent=2))
        print("Document:")
        print(results["documents"][i])


# ---------------------------
# 4. 建立 Embedding Model (Apple MPS / CPU)
# 由於本帥哥 我 是在 Mac M1 上做開發測試，因此使用MPS加速向量編碼
# ---------------------------
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("\nUsing device:", device)
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


# ---------------------------
# 5. 效能測試（Embedding + Query）
# ---------------------------
def benchmark_search(model, query_text, top_k=3):
    print("\n=== 效能測試 ===")
    print("Query text:", query_text)

    # Embedding 時間
    start = time.time()
    query_emb = model.encode(query_text).tolist()
    embedding_time = time.time() - start

    # Query 時間
    start = time.time()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["metadatas"]
    )
    query_time = time.time() - start

    print(f"Embedding time: {embedding_time:.4f} 秒")
    print(f"Query time: {query_time:.4f} 秒")

    return results


# ---------------------------
# 6. Main
# ---------------------------
if __name__ == "__main__":
    show_collections()
    show_collection_info()
    preview_documents(limit=5)

    model = load_model()
    benchmark_search(model, query_text="醫療保險") # 測試text，可以隨意更改
