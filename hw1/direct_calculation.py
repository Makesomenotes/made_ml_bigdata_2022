import numpy as np
import pandas as pd

df = pd.read_csv('/Users/maksimgritskikh/MADE/made_ml_bigdata_2022/hw1/AB_NYC_2019.csv')
# price = df['price'].dropna()
# print(np.mean(price), np.std(price) ** 2)
print(df['price'].isna())
print(df['price'].isna().sum())
