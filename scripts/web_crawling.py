"""
web_crawling.py

Generic web crawler for product pages.
This module demonstrates how to crawl product listings,
parse structured rules, and prepare data for downstream RAG pipelines.

⚠️ Disclaimer:
This code is for educational and research purposes only.
Users are responsible for ensuring that crawling targets comply
with the website's Terms of Service and robots.txt.
"""

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import time
import os
from typing import List, Dict, Any

# ================== Config ==================
BASE_URL = os.getenv("CRAWL_BASE_URL", "").rstrip("/")
LIST_PATH = os.getenv("CRAWL_LIST_PATH", "")
LIST_URL = f"{BASE_URL}{LIST_PATH}" if BASE_URL and LIST_PATH else ""

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EducationalCrawler/1.0)"
}

REQUEST_TIMEOUT = 15
REQUEST_DELAY = 0.2  # seconds

# ================== Utils ==================
def fetch_html(url: str) -> str:
    if not url:
        raise ValueError("Target URL is empty. Please set CRAWL_BASE_URL and CRAWL_LIST_PATH.")

    res = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    res.raise_for_status()
    return res.text


def html_br_to_newline(tag) -> str:
    for br in tag.find_all("br"):
        br.replace_with("\n")
    return tag.get_text("\n", strip=True)

# ================== Parse age by term ==================
def parse_age_by_term(text: str) -> List[Dict[str, Any]]:
    results = []
    if not text:
        return results

    lines = [l.strip() for l in re.split(r'[\n;；,，]+', text) if l.strip()]
    for ln in lines:
        m = re.search(r'(\d+)年期[:：]\s*(\d+)歲[~～\-](\d+)歲', ln)
        if m:
            results.append({
                "term": f"{m.group(1)}Y",
                "min_age": int(m.group(2)),
                "max_age": int(m.group(3))
            })
    return results

# ================== Parse amount rules ==================
def parse_amount_rules(text: str) -> List[Dict[str, Any]]:
    results = []
    if not text:
        return results

    lines = [l.strip() for l in re.split(r'[\n；;。]+', text) if l.strip()]
    for ln in lines:
        min_age, max_age = None, None
        min_amount, max_amount = None, None

        if "未達" in ln:
            m = re.search(r'未達\s*(\d+)', ln)
            if m:
                max_age = int(m.group(1)) - 1
                min_age = 0

        elif "以上" in ln:
            m = re.search(r'(\d+)', ln)
            if m:
                min_age = int(m.group(1))
                max_age = 99

        else:
            m = re.search(r'(\d+)\s*歲\s*[~～\-]\s*(\d+)\s*歲', ln)
            if m:
                min_age = int(m.group(1))
                max_age = int(m.group(2))

        m_amt = re.search(r'最低\s*(\d+)\s*萬(?:元)?\D*最高\s*(\d+)\s*萬', ln)
        if not m_amt:
            nums = re.findall(r'(\d+)\s*萬', ln)
            if len(nums) >= 2:
                min_amount = int(nums[0]) * 10000
                max_amount = int(nums[1]) * 10000

        if m_amt:
            min_amount = int(m_amt.group(1)) * 10000
            max_amount = int(m_amt.group(2)) * 10000

        if min_age is None:
            min_age = 0
        if max_age is None:
            max_age = 99

        if min_amount or max_amount:
            results.append({
                "min_age": min_age,
                "max_age": max_age,
                "min_amount": min_amount,
                "max_amount": max_amount,
                "raw_text": ln
            })

    return results

# ================== Crawl detail ==================
def crawl_product_detail(detail_url: str) -> Dict[str, Any]:
    html = fetch_html(detail_url)
    soup = BeautifulSoup(html, "html.parser")

    out = {
        "detail_url": detail_url,
        "age_raw_text": "",
        "age_parsed": [],
        "amount_raw_text": "",
        "amount_rules": []
    }

    titles = soup.find_all("div", class_="c-article-title")
    for t in titles:
        key = t.get_text(strip=True)
        nxt = t.find_next_sibling()
        if not nxt:
            continue

        if key in ["承保年齡", "投保年齡"]:
            out["age_raw_text"] = html_br_to_newline(nxt)
            out["age_parsed"] = parse_age_by_term(out["age_raw_text"])

        elif key in ["保額限制", "投保金額限制"]:
            out["amount_raw_text"] = html_br_to_newline(nxt)
            out["amount_rules"] = parse_amount_rules(out["amount_raw_text"])

    return out

# ================== Enrich product ==================
def enrich_product(product: Dict[str, Any]) -> Dict[str, Any]:
    detail = product.get("url")
    if not detail:
        return product

    detail_url = detail
    if detail.startswith("/") and BASE_URL:
        detail_url = BASE_URL + detail

    try:
        detail_data = crawl_product_detail(detail_url)
        product.update(detail_data)
    except Exception as e:
        product["_detail_error"] = str(e)

    return product

# ================== Main ==================
def main():
    if not LIST_URL:
        raise RuntimeError(
            "Missing configuration. Please set CRAWL_BASE_URL and CRAWL_LIST_PATH."
        )

    html = fetch_html(LIST_URL)
    soup = BeautifulSoup(html, "html.parser")

    cards = soup.find_all("div", class_="c-prodcard")
    print(f"Found {len(cards)} products")

    results = []

    for card in cards:
        data = {
            "title": "",
            "description": "",
            "url": ""
        }

        title_tag = card.find("h2", class_="c-prodcard-title")
        desc_tag = card.find("h3", class_="c-prodcard-descr")

        if title_tag:
            data["title"] = title_tag.get_text(strip=True)
        if desc_tag:
            data["description"] = desc_tag.get_text(strip=True)

        a_tag = card.find("a", href=True)
        if a_tag:
            data["url"] = a_tag["href"]

        labels = card.find_all("h5", class_="c-prodcard-detail-label")
        values = card.find_all("div", class_="c-prodcard-detail-cont")

        for l, v in zip(labels, values):
            key = l.get_text(strip=True)
            val = v.get_text(" ", strip=True)

            if key == "承保年齡":
                data["insured_age_label"] = val
            elif key == "繳費年期":
                data["payment_term"] = val
            elif key == "給付項目":
                data["benefits"] = val

        data = enrich_product(data)
        results.append(data)
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(results)

    keep_cols = [
        "title",
        "description",
        "url",
        "insured_age_label",
        "payment_term",
        "benefits",
        "age_raw_text",
        "amount_raw_text"
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

    output_file = "products_output.xlsx"
    df.to_excel(output_file, index=False)

    print(f"success exported: {output_file}")
    print(df.head())

if __name__ == "__main__":
    main()
