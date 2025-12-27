import polars as pl
import numpy as np

def map_at_k(actual, predicted, k=12):
    scores = []
    for a, p in zip(actual, predicted):
        if not a:
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

def run_baseline():
    # Top 12 items from the week before validation
    train = pl.read_parquet("train_split.parquet")
    last_week_start = pl.date(2020, 9, 9)
    
    top_12 = (
        train.filter(pl.col("t_dat") >= last_week_start)
        .group_by("article_id")
        .count()
        .sort("count", descending=True)
        .limit(12)
        .select("article_id")
        .to_series()
        .to_list()
    )
    
    #Ground Truth (What people actually bought in the val week)
    valid = pl.read_parquet("valid_split.parquet")
    actuals = (
        valid.group_by("customer_id")
        .agg(pl.col("article_id").alias("actual"))
    )
    
    # Evaluation
    preds = [top_12] * actuals.height
    score = map_at_k(actuals["actual"].to_list(), preds)
    
    print(f"--- Phase 1 Result ---")
    print(f"Popularity Baseline MAP@12: {score:.6f}")

if __name__ == "__main__":
    run_baseline()