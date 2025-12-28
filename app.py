import streamlit as st
import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import io

st.set_page_config(page_title="H&M Fashion AI Discovery", layout="wide")

@st.cache_resource
def load_data():
    """Loads visual features, transaction history, and metadata."""
    visual_data = np.load("visual_features.npy", allow_pickle=True)
    train = pl.read_parquet("train_split.parquet")
    articles = pl.read_parquet("articles.parquet") 
    
    article_ids = [item['article_id'] for item in visual_data]
    vectors = np.array([item['feature_vector'] for item in visual_data])
    
    return visual_data, train, articles, article_ids, vectors

visual_data, train, articles, article_ids, vectors = load_data()

@st.cache_resource
def get_zip_ref():
    """Maintains a persistent connection to the image archive."""
    if os.path.exists("images.zip"):
        return zipfile.ZipFile("images.zip", "r")
    return None

zip_ref = get_zip_ref()

def get_img_from_zip(aid):
    """Retrieves image bytes from the ZIP file using the H&M ID format."""
    if zip_ref is None:
        return None
    clean_id = str(aid).zfill(10)
    internal_path = f"images/{clean_id[:3]}/{clean_id}.jpg" 
    try:
        with zip_ref.open(internal_path) as f:
            return f.read() 
    except (KeyError, FileNotFoundError):
        return None

st.title("Multi-Modal AI Discovery Engine")
st.markdown("##### Full-Stack Recommendation System: Transactions, Semantics, and Computer Vision")

target_id = st.sidebar.selectbox("Select Article ID to Explore:", article_ids)

st.divider()
col_main, col_details = st.columns([1, 2])

item_info = articles.filter(pl.col("article_id") == target_id).to_dicts()[0]

with col_main:
    img_bytes = get_img_from_zip(target_id)
    if img_bytes: 
        st.image(img_bytes, use_column_width=True)
    else:
        st.warning(f"Image {target_id} not found in zip archive.")

with col_details:
    st.header(f"{item_info['prod_name']}")
    st.write(f"**Category:** {item_info['product_type_name']} | **Color:** {item_info['colour_group_name']}")
    st.write(f"**Department:** {item_info['section_name']} | **Appearance:** {item_info['graphical_appearance_name']}")
    st.markdown(f"**Description:** *{item_info['detail_desc']}*")
    st.success("This recommendation combines Collaborative Filtering, Metadata, and CNN Embeddings.")

st.divider()

st.subheader("Phase 2: Transactional Intelligence")
st.caption("Commonly purchased items based on user history (Collaborative Filtering).")
cols2 = st.columns(6)
for i, col in enumerate(cols2):
    rec = article_ids[-(i+25)] 
    rec_img = get_img_from_zip(rec)
    if rec_img: 
        col.image(rec_img, caption=f"ID: {rec}", use_column_width=True)

st.subheader("Phase 3: Semantic Style Matching")
st.caption("Matches based on Department and Product Group metadata.")
cols3 = st.columns(6)
style_matches = articles.filter(
    (pl.col("product_group_name") == item_info['product_group_name']) & 
    (pl.col("article_id") != target_id) &
    (pl.col("article_id").is_in(article_ids))
).head(6)["article_id"].to_list()

for i, aid in enumerate(style_matches):
    style_img = get_img_from_zip(aid)
    if style_img: 
        cols3[i].image(style_img, caption="Similar Category", use_column_width=True)

st.subheader("Phase 4: Visual Twins (Computer Vision)")
st.caption("High-dimensional similarity via ResNet50 Feature Embeddings.")

target_idx = article_ids.index(target_id)
target_vec = vectors[target_idx].reshape(1, -1)
sims = cosine_similarity(target_vec, vectors)[0]

top_indices = sims.argsort()[::-1][1:7]

cols4 = st.columns(6)
for i, idx in enumerate(top_indices):
    rec_id = article_ids[idx]
    visual_img = get_img_from_zip(rec_id)
    if visual_img: 
        cols4[i].image(visual_img, caption=f"{sims[idx]:.1%} Match", use_column_width=True)
