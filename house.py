import pandas as pd 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
d=pd.read_csv("Weather.csv")
print(d)
plt.xlabel("area")
plt.ylabel("price")
plt.scatter(d.area,d.price,color="darkblue",marker="*")
plt.show()
n_d=d.drop("price",axis="columns")
print(n_d)
p=d.price
print(p)
reg=linear_model.LinearRegression()
reg.fit(n_d,p)
print(reg.predict([[3300]]))
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[5000]]))
a_d=pd.read_csv("area.csv")
print(a_d)
pr=reg.predict(a_d)
print(pr)
a_d["prices"]=pr
print(a_d)
a_d.to_csv("prediction.csv")
plt.xlabel("area")
plt.ylabel("price")
plt.scatter(a_d.area,a_d.prices,color="darkblue",marker="*")
plt.show()