import polars as pl

train = pl.read_parquet("train_split.parquet")
articles = pl.read_parquet("articles_style.parquet")
top_12_popular = ['0706016001', '0706016002', '0372860001', '0610776002', '0759871002', 
                  '0464297007', '0720124001', '0156231001', '0547780003', '0156221001', 
                  '0610776001', '0399223001']

last_purchase = (
    train.sort("t_dat", descending=True)
    .group_by("customer_id")
    .first() # the single most recent transaction
    .select(["customer_id", "article_id"])
)

last_purchase_style = last_purchase.join(articles, on="article_id", how="left")

def find_similar_items(row):
    similar = (
        articles.filter(
            (pl.col("department_name") == row["department_name"]) & 
            (pl.col("colour_group_name") == row["colour_group_name"])
        )
        .limit(12)
        .get_column("article_id")
        .to_list()
    )
    return similar if len(similar) > 0 else top_12_popular

style_map = (
    articles.group_by(["department_name", "colour_group_name"])
    .agg(pl.col("article_id").alias("similar_items").head(12))
)

preds = (
    last_purchase_style.join(style_map, on=["department_name", "colour_group_name"], how="left")
    .select(["customer_id", "similar_items"])
)

print("Content-based recommendations")