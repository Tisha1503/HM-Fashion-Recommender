import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity

visual_data = np.load("visual_features.npy", allow_pickle=True)
train = pl.read_parquet("train_split.parquet")

article_ids = [item['article_id'] for item in visual_data]
vectors = np.array([item['feature_vector'] for item in visual_data])

print(f"Loaded {len(article_ids)} visual fingerprints.")

sim_matrix = cosine_similarity(vectors)

def get_visually_similar(target_id, n=6):
    if target_id not in article_ids:
        return []
    
    idx = article_ids.index(target_id)
    
    scores = list(enumerate(sim_matrix[idx]))
    
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    return [article_ids[i] for i, score in sorted_scores]

visual_recs_map = {}
for aid in article_ids:
    visual_recs_map[aid] = get_visually_similar(aid)

print("Visual similarity map created. You can now use this to boost your Hybrid model.")

test_id = article_ids[0]
print(f"Items that look like {test_id}: {visual_recs_map[test_id]}")