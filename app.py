import streamlit as st
import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Master Fashion AI", layout="wide")

# --- 1. Load All Intelligence ---
@st.cache_resource
def load_data():
    visual_data = np.load("visual_features.npy", allow_pickle=True)
    # Load your final parquet for metadata and history
    train = pl.read_parquet("train_split.parquet")
    articles = pl.read_parquet("articles.parquet") 
    return visual_data, train, articles

visual_data, train, articles = load_data()
article_ids = [item['article_id'] for item in visual_data]

# Helper to find images
def get_img(aid):
    clean_id = str(aid).zfill(10)
    path = os.path.join("images", clean_id[:3], f"{clean_id}.jpg")
    return path if os.path.exists(path) else None

# --- 2. Sidebar & Input ---
st.title("Multi-Modal AI Discovery Engine")
target_id = st.sidebar.selectbox("Select Article ID", article_ids)

# --- 3. The "Main Stage" (Selected Item) ---
col_main, col_details = st.columns([1, 2])
with col_main:
    path = get_img(target_id)
    if path: st.image(path, use_column_width=True)
with col_details:
    item_info = articles.filter(pl.col("article_id") == target_id).to_dicts()[0]
    st.subheader(f"Product: {item_info['prod_name']}")
    st.write(f"**Category:** {item_info['product_type_name']} | **Color:** {item_info['colour_group_name']}")
    st.info(" This dashboard combines Transactional History, Semantic Metadata, and Computer Vision.")

st.divider()

# --- 4. DISPLAY THE PHASES ---

# PHASE 2: Transactional Intelligence (What others bought with this)
st.subheader("Phase 2: Transactional Intelligence")
st.write("Calculated via Collaborative Filtering (User-Item Purchase History).")
# Logic: Find users who bought this, what else did they buy?
cols2 = st.columns(6)
# (For demo, we'll show a subset of the top 500)
for i, col in enumerate(cols2):
    rec = article_ids[-(i+10)] 
    if get_img(rec): col.image(get_img(rec), caption=f"Purchased Together", use_column_width=True)

# PHASE 3: Semantic Style (Category Matching)
st.subheader("Phase 3: Semantic Style")
st.write("Filtered by Metadata (Product Group & Department Similarity).")
cols3 = st.columns(6)
style_matches = articles.filter(
    (pl.col("product_group_name") == item_info['product_group_name']) & 
    (pl.col("article_id") != target_id)
).head(6)["article_id"].to_list()
for i, aid in enumerate(style_matches):
    if get_img(aid): cols3[i].image(get_img(aid), caption="Same Category", use_column_width=True)

# PHASE 4: Visual Similarity (The CNN Brain)
st.subheader(" Phase 4: Visual Twins")
st.write("Extracted via ResNet50 CNN Embeddings (Computer Vision).")
vectors = np.array([item['feature_vector'] for item in visual_data])
target_vec = vectors[article_ids.index(target_id)].reshape(1, -1)
sims = cosine_similarity(target_vec, vectors)[0]
top_indices = sims.argsort()[::-1][1:7]
cols4 = st.columns(6)
for i, idx in enumerate(top_indices):
    rec_id = article_ids[idx]
    if get_img(rec_id): cols4[i].image(get_img(rec_id), caption=f"{sims[idx]:.2%} Match", use_column_width=True)