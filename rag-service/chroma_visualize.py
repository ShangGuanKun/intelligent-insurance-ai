import warnings
import time
import matplotlib.pyplot as plt
import umap
import numpy as np
from chromadb import HttpClient
from tqdm import tqdm

# 關閉 UMAP 平行警告
warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value")

# ---------------------------
# 1. 連線 Chroma
# ---------------------------
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "insurance_products"

client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(COLLECTION_NAME)

# ---------------------------
# 2. 取得資料
# ---------------------------
print("Loading data from ChromaDB...")

results = collection.get(include=["embeddings", "metadatas"])

embeddings = np.array(results["embeddings"])
metadatas = results["metadatas"]

total = len(embeddings)
print(f"Total vectors: {total}")

# ---------------------------
# 3. 進度條 + UMAP 降維
# ---------------------------
print("Reducing dimensions (UMAP)...")

reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.05,
    n_components=2,
    metric="cosine",
    random_state=42
)

# 假進度條（視覺化使用者體驗）
with tqdm(total=100, desc="UMAP Progress") as pbar:
    for i in range(5):
        time.sleep(0.2)
        pbar.update(10)

    points_2d = reducer.fit_transform(embeddings)

    for i in range(5):
        time.sleep(0.2)
        pbar.update(10)

# ---------------------------
# 4. 建立分類
# ---------------------------
def get_label(meta: dict):
    if "target_group" in meta and meta["target_group"]:
        return meta["target_group"]
    elif "title" in meta:
        return meta["title"]
    return "未分類"


labels = [get_label(m) for m in metadatas]

unique_labels = list(set(labels))
color_map = {label: i for i, label in enumerate(unique_labels)}
colors = [color_map[label] for label in labels]

# ---------------------------
# 5. 繪圖
# ---------------------------
plt.figure(figsize=(12, 8))

plt.scatter(
    points_2d[:, 0],
    points_2d[:, 1],
    c=colors,
    cmap="tab20",
    s=25,
    alpha=0.75
)

plt.title("ChromaDB insurance products - Vector distribution diagram (UMAP)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")

plt.tight_layout()
plt.show()
