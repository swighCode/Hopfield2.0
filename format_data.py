import pandas as pd

df = pd.read_csv("data/results_pivot.csv")
# Pick every ~10th column, including 'method'
selected_cols = ['method'] + df.columns[1::len(df.columns)//10].tolist()
df_sampled = df[selected_cols]
df_sampled.to_csv("results_table.csv", index=False)
