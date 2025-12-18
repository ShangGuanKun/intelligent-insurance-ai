# Intelligent Insurance Consultation AI System

An end-to-end **AI-powered insurance consultation system** that combines:

* **LLM-based slot filling & conversation orchestration**
* **ML-based insurance premium prediction (XGBoost)**
* **RAG-based product recommendation using ChromaDB**

This project is designed as a **microservice-based architecture** and is suitable as a public GitHub portfolio project.

---

## ✨ Key Features

* **Conversational Insurance Consultation**
  Uses an LLM to guide users through structured slot filling.

* **Premium Prediction (ML Service)**
  Predicts insurance premium based on user features using a trained XGBoost model.

* **Product Recommendation (RAG Service)**
  Retrieves relevant insurance products and explanations via ChromaDB + embeddings.

* **Scalable Microservice Design**
  Each service is independently deployable and maintainable.

---

## 🏗️ System Architecture

![Architecture](images/RAG.jpeg)

**Services Overview**:

| Service      | Description                       | Port |
| ------------ | --------------------------------- | ---- |
| Frontend     | React-based UI                    | 3000 |
| Orchestrator | LLM orchestration & slot filling  | 5002 |
| ML Service   | Premium prediction (XGBoost)      | 5001 |
| RAG Service  | Product recommendation (ChromaDB) | 5003 |
| ChromaDB     | Vector database                   | 8000 |

---

## 📂 Project Structure

```
.
├── frontend/          # React frontend
├── orchestrator/      # LLM orchestration service (Flask)
├── ml-service/        # ML prediction service (Flask + XGBoost)
├── rag-service/       # RAG recommendation service (Flask + ChromaDB)
├── data/              # Sample input data
├── dataset/           # Training datasets
├── models/            # Trained ML models
├── scripts/           # Utility & test scripts
├── images/            # Architecture & demo screenshots
└── README.md
```

---

## 🧰 Tech Stack

* **Language**: Python 3.12
* **LLM**: Llama 3.1 (via local or API-based inference)
* **RAG**: ChromaDB + Sentence Transformers
* **ML**: XGBoost, scikit-learn
* **Backend**: Flask
* **Frontend**: React
* **Env Management**: `uv` (recommended)

---

## ⚙️ Environment Setup (Recommended)

This project **recommends using `uv`** for fast and clean dependency management.

### 1️⃣ Create Virtual Environment

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
uv pip install -r ml-service/requirements.txt
uv pip install -r rag-service/requirements.txt
uv pip install -r orchestrator/requirements.txt
```

---

## ▶️ Run Services

> Open separate terminals for each service.

### ML Service (Premium Prediction)

```bash
cd ml-service
python flask_predict_price.py
```

### RAG Service (Recommendation)

```bash
cd rag-service
python recommendation_service.py
```

### Orchestrator (LLM & Conversation)

```bash
cd orchestrator
python chat_with_llama.py
```

### Frontend

```bash
cd frontend
npm install
npm start
```

---

## 📊 Demo

Sample screenshots and results can be found in the `images/` directory, including:

* Slot filling conversation flow
* Premium prediction results
* RAG-based recommendation outputs

---

## 📌 Notes

* This repository **does not include runtime artifacts** such as virtual environments or ChromaDB persistence directories.
* `torch` is intentionally **not version-pinned** to support different OS / CPU / CUDA environments.
* The trained ML model is provided for demo purposes.

---

## 🚀 Future Improvements

* Docker & docker-compose support
* Async orchestration between ML and RAG services
* Streaming LLM responses
* Production-grade logging & monitoring

---

## 📜 License

This project is released for **educational and portfolio purposes**.

