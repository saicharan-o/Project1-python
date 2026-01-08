import pandas as pd
import matplotlib.pyplot as plt
d=pd.read_csv("TT.csv")
print(d)
plt.scatter(d["Mileage"],d["Sell Price($)"])
plt.show()
plt.scatter(d["Age(yrs)"],d["Sell Price($)"])
plt.show()
X=d[["Mileage","Age(yrs)"]]
y=d["Sell Price($)"]
from sklearn.model_selection import train_test_split
X_tr,X_t,y_tr,y_t=train_test_split(X,y,test_size=0.3)
print(X_tr)
print(X_t)
print(y_tr)
print(y_t)
from sklearn.linear_model import LinearRegression
c=LinearRegression()
c.fit(X_tr,y_tr)
print(X_tr)
print(c.predict(X_t))
print(y_t)
print(c.score(X_t,y_t))
X_tr,X_t,y_tr,y_t=train_test_split(X,y,test_size=0.3,random_state=10)
print(X_t)
