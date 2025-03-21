import pandas as pd

# Load dataset
df = pd.read_csv("data/postings.csv")

# Print basic info
print(df.head())
print(df.info())

