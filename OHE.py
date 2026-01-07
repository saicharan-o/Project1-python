import pandas as pd
d=pd.read_csv("Price_H.csv")
print(d)
dum=pd.get_dummies(d.town)
print(dum)
m=pd.concat([d,dum],axis="columns")
print(m)
f=m.drop(["town"],axis="columns")
print(f)
f=f.drop(["west windsor"],axis="columns")
print(f)
x=f.drop(["price"],axis="columns")
print(x)
y=f.price
print(y)
from sklearn import linear_model
mod=linear_model.LinearRegression()
mod.fit(x,y)
print(mod.predict(x))
print(mod.score(x,y))
print(mod.predict([[3800,2,1]]))
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df=d
df.town=l.fit_transform(df.town)
print(df)
X=df[["town","area"]].values
print("New X value \n",X)
y=df.price.values
print("New Y value \n",y)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
c=ColumnTransformer([("town",OneHotEncoder(),[0])],remainder="passthrough")
x=c.fit_transform(X)
print("New modified x value \n",x)
x=x[:,1:]
print("New modified version x value \n",x)
mod.fit(x,y)
print("Prediction 1: ",mod.predict([[2,1,4000]]))
print("Prediction 2: ",mod.predict([[0,2,3000]]))
print("Prediction 3: ",mod.predict([[3,2,4500]]))