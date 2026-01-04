import pandas as pd 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
d=pd.read_csv("House-price.csv")
print(d)
print(d.bedrooms.median())
d.bedrooms=d.bedrooms.fillna(d.bedrooms.median())
print(d)
reg=linear_model.LinearRegression()
reg.fit(d.drop("price",axis="columns"),d.price)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(pd.DataFrame({"area":[2000],"bedrooms":[6],"age":[30]})))
print(reg.predict(pd.DataFrame({"area":[4200],"bedrooms":[4],"age":[50]})))