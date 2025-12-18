from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path

# ---------------------------
# Load Excel
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR.parent / "data" / "insurance_sample.csv"
df = pd.read_excel(data_path)

# Replace NaN in URL & title to empty strings
df["url"] = df["url"].fillna("").astype(str)
df["title"] = df["title"].fillna("").astype(str)
df["description"] = df["description"].fillna("").astype(str)
df["insured_age_label"] = df["insured_age_label"].fillna("").astype(str)
df["payment_term"] = df["payment_term"].fillna("").astype(str)
df["benefits"] = df["benefits"].fillna("").astype(str)
df["age_raw_text"] = df["age_raw_text"].fillna("").astype(str)
df["amount_raw_text"] = df["amount_raw_text"].fillna("").astype(str)

def build_doc(row):
    return f"""
商品名稱：{row['title']}
商品描述：{row['description']}
投保年齡：{row['insured_age_label']}
繳費期間：{row['payment_term']}
保障內容：{row['benefits']}
網址：{row['url']}
原始年齡資訊：{row['age_raw_text']}
保額說明：{row['amount_raw_text']}
""".strip()

df["doc"] = df.apply(build_doc, axis=1)

# ---------------------------
# Connect to Chroma HTTP Server (Docker)
# ---------------------------
client = HttpClient(host="localhost", port=8000)

# Create collection
collection_name = "insurance_products"
try:
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
except:
    collection = client.get_collection(collection_name)

# ---------------------------
# Embedding model
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Insert into ChromaDB (HTTP)
# ---------------------------
ids = []
documents = []
metadatas = []

for idx, row in df.iterrows():
    product_id = f"prod_{idx}"

    ids.append(product_id)
    documents.append(row["doc"])
    metadatas.append({
        "title": row["title"],
        "url": row["url"]
    })

embeds = model.encode(documents).tolist()

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeds
)

print("success write", len(df), "items into ChromaDB!")
