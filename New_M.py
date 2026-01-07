import pandas as pd 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
d=pd.read_csv("New_H.csv")
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
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[5000]]))
import  pickle
with open("model_F","wb") as file:
    pickle.dump(reg,file)
with open("model_F","rb") as file:
    m=pickle.load(file)
print(m.predict(pd.DataFrame({"area": [5500]})))    
import joblib
joblib.dump(reg,"model_J")
j=joblib.load("model_J")
print(j.predict(pd.DataFrame({"area":[6000]})))