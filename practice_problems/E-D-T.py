import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
d=pd.read_csv("Survival.csv")

print(d.head())

d=d[["Pclass","Sex","Age","Fare","Survived"]]
print(d)

d['Age'].fillna(d['Age'].median(),inplace=True)
d['Fare'].fillna(d['Fare'].median(),inplace=True)
d['Sex']=d['Sex'].map({"male":0,"female":1})
n=d[["Pclass","Sex","Age","Fare"]]
e=d["Survived"]

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_tr,X_t,y_tr,y_t=train_test_split(n,e,train_size=0.2)
m=DecisionTreeClassifier()
m.fit(X_tr,y_tr)
y_P=m.predict(X_t)

print(y_P)

print("\nScore of Model: ",m.score(X_t,y_t))

print("\nlength of X train : ",len(X_tr))

print("\n length of X test : ",len(X_t))

print("\nlength of Y train : ",len(y_tr))

print("\n length of Y test : ",len(y_t))


