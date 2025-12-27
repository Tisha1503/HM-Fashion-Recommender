import polars as pl

def create_split():
    df = pl.read_parquet("transactions.parquet")
    
    # Last date = 2020-09-22 
    val_start = pl.date(2020, 9, 16)
    # 1 week prior to last date for validation set( 9 to 15th)
    
    train = df.filter(pl.col("t_dat") < val_start)
    valid = df.filter(pl.col("t_dat") >= val_start)
    
    train.write_parquet("train_split.parquet")
    valid.write_parquet("valid_split.parquet")
    print(f"Split complete. Validation rows: {valid.height}")

if __name__ == "__main__":
    create_split()