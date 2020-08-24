# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:18:06 2020

@author: jiraw
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



DiscRatio = np.array([0, 0.3, 0.5])
Sat = np.array([0.85, 0.73, 0.60])
Sun = np.array([0.16, 0.58, 1.30])
Mon = np.array([1.64, 1.42, 1.10])


data = np.array([[DiscRatio, Sat, Mon]])
data = data.reshape(-1, 3)
data_T = np.transpose(data)

dis20 = (data_T[0] + (data_T[1] - data_T[0])*2/3)
dis40 = (data_T[1] + (data_T[2] - data_T[1])*1/2)


regressor = LinearRegression()
regressor.fit(data, Sun.reshape(-1,1))
print("Sun for Discount 20%: ")
print(regressor.predict(dis20.reshape(-1,3)))
print("Sun for Discount 40%: ") 
print(regressor.predict(dis40.reshape(-1,3)))
"""
xx = np.linspace(0.7, 0.5)
yy = np.linspace(0.1, 1.4)
zz = np.linspace(1.8, 1.0)
#zz = np.linspace(0, 0.5)
grid = (xx, yy, zz)
grid = np.transpose(grid)
fig = plt.figure()
ax=plt.axes(projection='3d')
for ratio in data:
    ax.scatter(ratio[0], ratio[1], ratio[2])
ax.plot3D(xx, yy, zz)
"""