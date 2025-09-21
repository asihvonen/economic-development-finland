import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_excel('Household income and expenditure.xlsx', index_col=False)

rows_per_region = 13

dfs = {}



for i in range(119):
    region = df.iloc[i*rows_per_region:(i+1)*rows_per_region]
    name = region['Unnamed: 0'].iloc[0]
    region = region.drop('Unnamed: 0', axis=1)
    region = region.T
    new_header = region.iloc[0]
    region = region[1:]
    region.columns = new_header
    dfs[name] = region


plt.figure(figsize=(10, 6))
plt.plot(dfs['WHOLE COUNTRY'].index, dfs['WHOLE COUNTRY']['B2N Operating surplus, net'])
plt.title('B1GMH Over Time')
plt.ylabel('Value')
plt.xlabel('Year')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.tight_layout()
plt.show()

