import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm
from statsmodels.api import OLS
from sklearn.preprocessing import PolynomialFeatures
# %matplotlib inline

hubway_data = pd.read_csv('D:\zhamppx\data_set_lab4\dataset_1_test.txt', low_memory=False)
hubway_data.head()

time_min = hubway_data['TimeMin']
pick = hubway_data['PickupCount']

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.scatter(time_min, pick)

ax.set_xlabel('TimeMin')
ax.set_ylabel('PickupCount')
ax.set_title('Dataset 1 Test')
ax.legend()

plt.show()
