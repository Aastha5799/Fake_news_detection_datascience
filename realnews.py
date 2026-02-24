import pandas as pd

df = pd.read_csv("train.csv")
print(df["label"].value_counts())  # Check distribution of real vs. fake news

