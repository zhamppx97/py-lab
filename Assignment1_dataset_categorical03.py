import pandas as pd
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import datetime
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set(style="ticks")

df = pd.read_csv('D:\\zhamppx\\assignment\\dataset_categorical03.csv', low_memory = False)
df.head()

fig, ax = plt.subplots()
ds = df.apply(pd.Series.value_counts).sum(axis = 1, skipna = True)

print('\ndataset categorical 03 ploting\n')
print(ds)

ds.plot(kind='bar')
ax.set_title('dataset categorical 03 ploting')
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N', 'Y', ' '], rotation=0)
plt.ylabel('Count', fontsize=10)
plt.xlabel('Values', fontsize=10)
plt.show()
