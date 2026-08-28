import matplotlib.pyplot as plt 
import pandas as pd

d=pd.read_csv("HR.csv")
print(d.head())

l=d[d.left==1]
print(l)

r=d[d.left==0]
print(r)
print(l.shape)
print(r.shape)
print(d.groupby("left").mean(numeric_only=True))
pd.crosstab(d.salary,d.left).plot(kind="bar")
pd.crosstab(d.Department,d.left).plot(kind="bar")
df=d[["satisfaction_level","average_montly_hours","promotion_last_5years","salary"]]
print(df.head())
s=pd.get_dummies(df.salary,prefix="salary",dtype=int)
dm=pd.concat([df,s],axis="columns")
print(dm.head())
dm=dm.drop("salary",axis="columns")
print(dm)
x=dm
print(dm)
y=d.left
from  sklearn.model_selection import train_test_split
X_tr,X_t,y_tr,y_t=train_test_split(x,y,train_size=0.7,random_state=42)
from sklearn.linear_model import LogisticRegression
m=LogisticRegression(max_iter=1000)
m.fit(X_tr,y_tr)
print(m.predict(X_t))
print(m.score(X_t,y_t))
plt.show()