import polars as pl
import numpy as np

train = pl.read_parquet("train_split.parquet")
valid = pl.read_parquet("valid_split.parquet")
articles = pl.read_parquet("articles_style.parquet")

top_12_popular = (
    train.filter(pl.col("t_dat") >= pl.date(2020, 9, 9))
    .group_by("article_id").len()
    .sort("len", descending=True).limit(12)
    .select("article_id").to_series().to_list()
)

user_history = (
    train.sort("t_dat", descending=True)
    .group_by("customer_id")
    .agg(pl.col("article_id").unique(maintain_order=True).head(6))
)

last_purchase = train.sort("t_dat", descending=True).group_by("customer_id").first()
last_style = last_purchase.join(articles, on="article_id", how="left")

# Create a map of Style 
style_map = (
    articles.group_by(["department_name", "colour_group_name"])
    .agg(pl.col("article_id").alias("style_candidates").head(6))
)

user_style_recs = last_style.join(style_map, on=["department_name", "colour_group_name"], how="left")

# 5. Merge Everything
print("Merging histories and style similarities...")
actuals = valid.group_by("customer_id").agg(pl.col("article_id").alias("actual"))

# Join all parts
final_df = (
    actuals
    .join(user_history, on="customer_id", how="left")
    .join(user_style_recs.select(["customer_id", "style_candidates"]), on="customer_id", how="left")
)

def hybrid_logic(row):
    # row[0] is actual, row[1] is history, row[2] is style_candidates
    hist = list(row[1]) if row[1] is not None else []
    style = list(row[2]) if row[2] is not None else []
    
    res = hist + [s for s in style if s not in hist]
    
    if len(res) < 12:
        res.extend([p for p in top_12_popular if p not in res])
        
    return res[:12]

final_df = final_df.with_columns(
    pl.struct(["actual", "article_id", "style_candidates"])
    .map_elements(
        lambda x: hybrid_logic(list(x.values())), # Convert dict_values to a list here
        return_dtype=pl.List(pl.String)
    )
    .alias("predictions")
)

# 6. Final Evaluation
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

score = map_at_k(final_df["actual"].to_list(), final_df["predictions"].to_list())
print(f"--- Phase 3 Result ---")
print(f"Master Hybrid MAP@12: {score:.6f}")