import polars as pl
import numpy as np

train = pl.read_parquet("train_split.parquet")
valid = pl.read_parquet("valid_split.parquet")

print("Step 1: Calculating Global Top 12 (The 'Fillers - if a customer bought less than 12 items in the given timeline')")
last_week_start = pl.date(2020, 9, 9)
top_12_popular = (
    train.filter(pl.col("t_dat") >= last_week_start)
    .group_by("article_id")
    .len()
    .sort("len", descending=True)
    .limit(12)
    .select("article_id")
    .to_series()
    .to_list()
)

user_history = (
    train.sort("t_dat", descending=True)
    .group_by("customer_id")
    .agg(pl.col("article_id").unique(maintain_order=True).head(12))
)

actuals = valid.group_by("customer_id").agg(pl.col("article_id").alias("actual"))

eval_df = actuals.join(user_history, on="customer_id", how="left")

def fill_recommendations(history):
    if history is None:
        return top_12_popular
    
    res = list(history)
    if len(res) < 12:
        for item in top_12_popular:
            if item not in res:
                res.append(item)
            if len(res) == 12:
                break
    return res[:12]

eval_df = eval_df.with_columns(
    pl.col("article_id").map_elements(fill_recommendations, return_dtype=pl.List(pl.String)).alias("predictions")
)

print("Step 4: Evaluating MAP@12...")
def map_at_k(actual, predicted, k=12):
    scores = []
    for a, p in zip(actual, predicted):
        if a is None or p is None:
            scores.append(0.0)
            continue
            
        score = 0.0
        num_hits = 0.0
        for i, p_item in enumerate(p[:k]):
            if p_item in a:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        scores.append(score / min(len(a), k))
    return np.mean(scores)

score = map_at_k(eval_df["actual"].to_list(), eval_df["predictions"].to_list())
print(f"Personalized Baseline MAP@12: {score:.6f}")