
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="B2B Category Mapping Validator", layout="wide")

st.title("🔍 B2B Category Mapping Validator (MVP)")

# Inputs
product_url = st.text_input("Enter Product URL", placeholder="https://yourb2b.com/product/123")
category_name = st.text_input("Enter Mapped Category Name", placeholder="Example: HDPE Plastic Crates")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Mock scraping function for MVP
def mock_scrape_product(url):
    return {
        "title": "HDPE Plastic Crate - 45L",
        "price": "₹180",
        "specs": ["Material: HDPE", "Capacity: 45L", "Color: Blue"],
        "image_tags": ["side_view", "top_view"]
    }

# Similarity scoring
def title_similarity(product_title, category_name):
    emb1 = model.encode(product_title, convert_to_tensor=True)
    emb2 = model.encode(category_name, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())

# Convert score to RAG
def get_rag(score):
    if score >= 0.7:
        return "🟢 Green"
    elif score >= 0.4:
        return "🟠 Amber"
    else:
        return "🔴 Red"

# Output builder
def build_output(product_info, category_name):
    title_score = title_similarity(product_info['title'], category_name)
    spec_match = len(product_info['specs']) / 5  # Assume 5 ideal specs
    image_score = len(product_info['image_tags']) / 3  # Assume 3 ideal images

    results = [
        {
            "Attribute": "Title",
            "Product Value": product_info['title'],
            "Score": round(title_score, 2),
            "RAG": get_rag(title_score),
            "Reason": "Similarity to category name"
        },
        {
            "Attribute": "Specifications",
            "Product Value": f"{len(product_info['specs'])} specs",
            "Score": round(spec_match, 2),
            "RAG": get_rag(spec_match),
            "Reason": "Specs match ratio (ideal = 5)"
        },
        {
            "Attribute": "Images",
            "Product Value": ", ".join(product_info['image_tags']),
            "Score": round(image_score, 2),
            "RAG": get_rag(image_score),
            "Reason": "Multi-angle image coverage (ideal = 3)"
        }
    ]

    return pd.DataFrame(results)

# Run if inputs given
if product_url and category_name:
    with st.spinner("Analyzing product and category..."):
        product_info = mock_scrape_product(product_url)
        df = build_output(product_info, category_name)
        st.success("Analysis complete.")
        st.dataframe(df, use_container_width=True)
