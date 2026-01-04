import pandas as pd 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
d=pd.read_csv("Canada.csv")
print(d)
plt.xlabel("year")
plt.ylabel("per_capita_income")
plt.scatter(d.year,d.per_capita_income,color="darkblue",marker="*")
plt.show()
n_d=d.drop("per_capita_income",axis="columns")
print(n_d)
p=d.per_capita_income
print(p)
reg=linear_model.LinearRegression()
reg.fit(n_d,p)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(pd.DataFrame({"year": [1960]})))
print(reg.predict(pd.DataFrame({"year": [2027]})))
y_d=pd.read_csv("Canada's-per.csv")
print(y_d)
per=reg.predict(y_d)
print(per)
y_d["per_capita_incomes"]=per
print(y_d)
y_d.to_csv("predictions.csv")
plt.xlabel("year")
plt.ylabel("per_capita_incomes")
plt.scatter(y_d.year,y_d.per_capita_incomes,color="darkblue",marker="*")
plt.show()