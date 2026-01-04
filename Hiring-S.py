import pandas as pd 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from word2number import w2n
import math
d=pd.read_csv("Hiring-M.csv")
print(d)
d.experience=d.experience.fillna("zero")
print(d)
d.experience=d.experience.apply(w2n.word_to_num)
print(d)
m=math.floor(d["test_score(out of 10)"].mean())
print(m)
d["test_score(out of 10)"]=d["test_score(out of 10)"].fillna(m)
print(d)
reg=linear_model.LinearRegression()
reg.fit(d.drop("salary($)",axis="columns"),d["salary($)"])
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(pd.DataFrame({"experience":[2],"test_score(out of 10)":[9],"interview_score(out of 10)":[6]})))
print(reg.predict(pd.DataFrame({"experience":[12],"test_score(out of 10)":[10],"interview_score(out of 10)":[10]})))
print(reg.predict(pd.DataFrame({"experience":[0],"test_score(out of 10)":[1],"interview_score(out of 10)":[2]})))
print(reg.predict(pd.DataFrame({"experience":[20],"test_score(out of 10)":[10],"interview_score(out of 10)":[10]})))
print(reg.predict(pd.DataFrame({"experience":[15],"test_score(out of 10)":[6],"interview_score(out of 10)":[8]})))