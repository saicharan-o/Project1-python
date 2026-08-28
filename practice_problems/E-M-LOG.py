from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

d=load_iris()
print(dir(d))

print("DESCR: \n",d.DESCR) 
print("DATA: \n",d.data) 
print("DATA_MODULE: \n",d.data_module) 
print("Feature_Names: \n",d.feature_names) 
print("Filename: \n",d.filename)  
print("Frame: \n",d.frame) 
print("Target: \n",d.target) 
print("Target_Names: \n",d.target_names)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

X_tr,X_t,y_tr,y_t=train_test_split(d.data,d.target,train_size=0.3)

m=LogisticRegression()
m.fit(X_tr,y_tr)
print(m.score(X_t,y_t))

y_P=m.predict(X_t)


from sklearn.metrics import accuracy_score

c=accuracy_score(y_t,y_P)
print(c)

s = X_t[:5]
p= m.predict(s)

for i, pred in enumerate(p):
    print(f"Sample {i+1} Prediction:", d.target_names[pred])
    
plt.scatter(X_t[:,0], X_t[:,1], c=y_P)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Prediction")
plt.show()
