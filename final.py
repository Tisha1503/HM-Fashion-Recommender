import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity

visual_data = np.load("visual_features.npy", allow_pickle=True)
train = pl.read_parquet("train_split.parquet")
valid = pl.read_parquet("valid_split.parquet")

article_ids = [item['article_id'] for item in visual_data]
vectors = np.array([item['feature_vector'] for item in visual_data])
sim_matrix = cosine_similarity(vectors)

visual_map = {}
for i, aid in enumerate(article_ids):
    idx_scores = sorted(list(enumerate(sim_matrix[i])), key=lambda x: x[1], reverse=True)[1:4]
    visual_map[aid] = [article_ids[idx] for idx, score in idx_scores]

print("Generating Final Recommendations...")

user_history = (
    train.sort("t_dat", descending=True)
    .group_by("customer_id")
    .agg(pl.col("article_id").unique(maintain_order=True).head(6))
    .rename({"article_id": "history"})
)

last_item = (
    train.sort("t_dat", descending=True)
    .group_by("customer_id")
    .first()
    .select([pl.col("customer_id"), pl.col("article_id").alias("last_bought_id")])
)

top_3_pop = train["article_id"].value_counts().sort("count", descending=True).head(3)["article_id"].to_list()

actuals = valid.group_by("customer_id").agg(pl.col("article_id").alias("actual"))

final_eval = (
    actuals
    .join(user_history, on="customer_id", how="left")
    .join(last_item, on="customer_id", how="left")
)

def final_logic(history, last_id):
    res = list(history) if history is not None else []
    
    if last_id is not None:
        clean_id = str(last_id).zfill(10)
        vis_matches = visual_map.get(clean_id, [])
        for v in vis_matches:
            if v not in res:
                res.append(v)
    
    for p in top_3_pop:
        if p not in res:
            res.append(p)
        if len(res) >= 12:
            break
            
    return res[:12]

final_eval = final_eval.with_columns(
    pl.struct(["history", "last_bought_id"])
    .map_elements(lambda x: final_logic(x["history"], x["last_bought_id"]), return_dtype=pl.List(pl.String))
    .alias("predictions")
)


def map_at_k(actual, predicted, k=12):
    scores = []
    for a, p in zip(actual, predicted):
        if a is None or p is None:
            scores.append(0.0)
            continue
        score, hits = 0.0, 0.0
        for i, p_item in enumerate(p[:k]):
            if p_item in a:
                hits += 1.0
                score += hits / (i + 1.0)
        scores.append(score / min(len(a), k))
    return np.mean(scores)

final_score = map_at_k(final_eval["actual"].to_list(), final_eval["predictions"].to_list())
print(f"\n---FINAL  RESULT ---")
print(f"Multi-Modal Hybrid MAP@12: {final_score:.6f}")