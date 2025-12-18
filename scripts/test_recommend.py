import requests

API_URL = "http://localhost:5003/recommend_products"

def test_recommend():
    payload = {
        "query": "我想找醫療保障高、保障內容全面、預算大約8萬元的保險方案",
        "top_k": 3 # 前3相關係數高的
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            data = response.json()
            print("\n--- Parsed Products ---")
            for i, p in enumerate(data.get("products", []), 1):
                print(f"\n[{i}] {p.get('title')}")
                print(f"     Score: {p.get('score')}")
                print(f"     URL: {p.get('url')}")
                print(f"     Summary:\n{p.get('summary')}")
        else:
            print("Error response:", response.text)

    except Exception as e:
        print("Exception:", e)

if __name__ == "__main__":
    test_recommend()
