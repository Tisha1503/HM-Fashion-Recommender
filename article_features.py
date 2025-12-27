import polars as pl

# Load metadata
articles = pl.read_parquet("articles.parquet")

# Select columns that define 'Style'
style_columns = [
    "product_type_name", 
    "product_group_name", 
    "colour_group_name", 
    "perceived_colour_value_name",
    "department_name"
]

# Create a simplified style profile
articles_style = articles.select(["article_id"] + style_columns)

articles_style.write_parquet("articles_style.parquet")
print("Phase 3.1: Article style profiles created.")