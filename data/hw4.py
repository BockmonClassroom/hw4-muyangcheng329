import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# read
df = pd.read_csv("Iris.csv")

features = df.columns[:-1]

# normalized
minmax_scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[features] = minmax_scaler.fit_transform(df[features])
df_normalized.to_csv("Normalized.csv", index=False)

# standardized
standard_scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[features] = standard_scaler.fit_transform(df[features])
df_standardized.to_csv("Standardized.csv", index=False)
