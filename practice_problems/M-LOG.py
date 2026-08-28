from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

d = load_digits()
plt.gray() 

for i in range(5):
    plt.matshow(d.images[i],cmap='jet')    

print(dir(d))
print("IMAGES: \n",d.images)
print("DESCR: \n",d.DESCR)
print("DATA: \n",d.data)
print("Feature_names: \n",d.feature_names)
print("FRAME: \n",d.frame) 
print("TARGET: \n",d.target)
print("Target_names: \n",d.target_names)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_tr,X_t,y_tr,y_t=train_test_split(d.data,d.target,train_size=0.3)
m=LogisticRegression()
m.fit(X_tr,y_tr)

print(m.score(X_t,y_t))

y_P=m.predict(X_t)

from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_t,y_P)
print(c)

import seaborn as sn

plt.figure(figsize = (10,7))
sn.heatmap(c, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()