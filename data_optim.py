import polars as pl

def optimize():
    print("Transaction optimization...")
    trans = pl.read_csv("transactions_train.csv").with_columns([
        pl.col("t_dat").str.to_date(),
        pl.col("article_id").cast(pl.Utf8),
        pl.col("price").cast(pl.Float32),
        pl.col("sales_channel_id").cast(pl.Int8)
    ])
    trans.write_parquet("transactions.parquet")

    print("Articles optimization...")
    articles = pl.read_csv("articles.csv", schema_overrides={"article_id": pl.String})
    articles.write_parquet("articles.parquet")

    print("Customers Optimization...")
    customers = pl.read_csv("customers.csv")
    customers.write_parquet("customers.parquet")
    print(".parquet files ready.")

if __name__ == "__main__":
    optimize()