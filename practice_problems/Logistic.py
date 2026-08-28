import pandas as pd
import matplotlib.pyplot as plt

d=pd.read_csv("Insurance.csv")
print(d)

plt.scatter(d.age,d.bought_insurance,marker='+',color='black')
plt.show()

from sklearn.model_selection import train_test_split
X_tr,X_t,y_tr,y_t=train_test_split(d[["age"]],d.bought_insurance,train_size=0.8,random_state=42)
print(X_t)

from sklearn.linear_model import LogisticRegression
m=LogisticRegression()
m.fit(X_tr,y_tr)
print(X_t)

y_P=m.predict(X_t)

print(m.predict_proba(X_t))
print(m.score(X_t,y_t))
print(y_P)
print(X_t)
print(m.intercept_)
print(m.coef_)

import math
def predict_fun(age):
    z = m.coef_[0][0] * age + m.intercept_[0]
    return 1 / (1 + math.exp(-z))

age=30
print("Age: 30",predict_fun(age))

age=50
print("Age: 50",predict_fun(age))

age=0
print("Age: 0",predict_fun(age))
    
age=10
print("Age: 10",predict_fun(age)) 