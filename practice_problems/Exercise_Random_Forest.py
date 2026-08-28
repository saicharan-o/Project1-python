import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("Exercise_Random_Forest.csv")

le_brand = LabelEncoder()
le_motor = LabelEncoder()
df["Brand_enc"] = le_brand.fit_transform(df["Brand"])
df["Motor_enc"] = le_motor.fit_transform(df["Motor_Type"])

X = df[["Brand_enc","Year","Battery_kWh","Range_miles","Top_Speed_mph","Acceleration_0_60","Motor_enc"]]
y = df["Fast_Charger_Compatible"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:  ", confusion_matrix(y_test, y_pred,labels=[0, 1]))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", importance)

plt.figure(figsize=(8,5))
plt.barh(importance["Feature"], importance["Importance"])
plt.title("Random Forest Feature Importance-Figure-1")
plt.gca().invert_yaxis()
plt.show()

df["Battery_kWh"].hist(bins=10)
plt.xlabel("Battery (kWh)")
plt.ylabel("Frequency")
plt.title("Battery Capacity Distribution-Figure-2")
plt.show()

df["Range_miles"].hist(bins=10)
plt.xlabel("Range (miles)")
plt.ylabel("Frequency")
plt.title("Range Distribution-Figure-3")
plt.show()

df["Top_Speed_mph"].hist(bins=10)
plt.xlabel("Top Speed (mph)")
plt.ylabel("Frequency")
plt.title("Top Speed Distribution-Figure-4")
plt.show()

df["Acceleration_0_60"].hist(bins=10)
plt.xlabel("Acceleration (0-60 sec)")
plt.ylabel("Frequency")
plt.title("Acceleration Distribution-Figure-5")
plt.show()

df["Brand"].value_counts().plot(kind="bar")
plt.xlabel("Brand")
plt.ylabel("Count")
plt.title("Number of Cars per Brand-Figure-6")
plt.show()

df.groupby(["Brand","Fast_Charger_Compatible"])["Model"].count().unstack().plot(kind="bar", stacked=True)
plt.ylabel("Number of Cars")
plt.title("Fast Charger Compatibility per Brand-Figure-7")
plt.show()

df["Fast_Charger_Compatible"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Fast Charger Compatibility Distribution-Figure-8")
plt.show()

plt.scatter(df["Battery_kWh"], df["Range_miles"])
plt.xlabel("Battery (kWh)")
plt.ylabel("Range (miles)")
plt.title("Battery vs Range-Figure-9")
plt.show()

plt.scatter(df["Top_Speed_mph"], df["Acceleration_0_60"])
plt.xlabel("Top Speed (mph)")
plt.ylabel("0-60 Acceleration (sec)")
plt.title("Top Speed vs Acceleration-Figure-10")
plt.show()

plt.scatter(df["Range_miles"], df["Acceleration_0_60"], c=df["Fast_Charger_Compatible"])
plt.xlabel("Range (miles)")
plt.ylabel("0-60 Acceleration (sec)")
plt.title("Range vs Acceleration (Color=Fast Charger)-Figure-11")
plt.colorbar(label="Fast Charger")
plt.show()

df.boxplot(column="Battery_kWh", by="Fast_Charger_Compatible")
plt.title("Battery by Fast Charger-Figure-12")
plt.suptitle("")
plt.ylabel("Battery (kWh)")
plt.show()

df.boxplot(column="Range_miles", by="Fast_Charger_Compatible")
plt.title("Range by Fast Charger-Figure-13")
plt.suptitle("")
plt.ylabel("Range (miles)")
plt.show()


import seaborn as sns

sns.violinplot(x="Fast_Charger_Compatible", y="Acceleration_0_60", data=df)
plt.title("Acceleration Distribution by Fast Charger-Figure-14")
plt.show()

sns.swarmplot(x="Brand", y="Top_Speed_mph", data=df)
plt.title("Top Speed Distribution per Brand-Figure-15")
plt.show()


from pandas.plotting import scatter_matrix

scatter_matrix(df[["Battery_kWh","Range_miles","Top_Speed_mph","Acceleration_0_60"]], figsize=(10,10))
plt.suptitle("Scatter Matrix of EV Features-Figure-16")
plt.show()

sns.heatmap(df[["Battery_kWh","Range_miles","Top_Speed_mph","Acceleration_0_60","Fast_Charger_Compatible"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap-Figure-17")
plt.show()

sns.pairplot(df, vars=["Battery_kWh","Range_miles","Top_Speed_mph","Acceleration_0_60"], hue="Fast_Charger_Compatible")
plt.suptitle("Pairwise Relationships of EV Performance Features (Hue: Fast Charger Compatible)-Figure-18",y=1.02)
plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["Battery_kWh"], df["Range_miles"], df["Top_Speed_mph"], c=df["Fast_Charger_Compatible"])
ax.set_xlabel("Battery")
ax.set_ylabel("Range")
ax.set_zlabel("Top Speed")
plt.title("3D Scatter: Battery vs Range vs Top Speed-Figure-19")
plt.show()

plt.scatter(df["Range_miles"], df["Top_Speed_mph"], s=df["Acceleration_0_60"]*50, c=df["Fast_Charger_Compatible"])
plt.xlabel("Range (miles)")
plt.ylabel("Top Speed (mph)")
plt.title("Range vs Top Speed (Size=Acceleration)-Figure-20")
plt.show()


import numpy as np

labels=np.array(["Battery","Range","Top Speed","Acceleration"])
stats=df.loc[0,["Battery_kWh","Range_miles","Top_Speed_mph","Acceleration_0_60"]].values

angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))
fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=2)
ax.fill(angles, stats, alpha=0.25)
ax.set_thetagrids(angles[:-1]*180/np.pi, labels)
plt.title("Radar Plot for Tesla Model S-Figure-21")
plt.show()

df.groupby("Motor_Type")["Battery_kWh"].mean().plot(kind="bar")
plt.title("Average Battery by Motor Type-Figure-22")
plt.ylabel("Battery (kWh)")
plt.show()

df.groupby("Motor_Type")["Range_miles"].mean().plot(kind="bar")
plt.title("Average Range by Motor Type-Figure-23")
plt.ylabel("Range (miles)")
plt.show()

plt.scatter(df["Battery_kWh"], df["Acceleration_0_60"], c=df["Fast_Charger_Compatible"])
plt.xlabel("Battery")
plt.ylabel("Acceleration (0-60)")
plt.title("Battery vs Acceleration (Colored by Charger)-Figure-24")
plt.show()

plt.scatter(df["Top_Speed_mph"], df["Battery_kWh"], c=df["Fast_Charger_Compatible"])
plt.xlabel("Top Speed")
plt.ylabel("Battery")
plt.title("Top Speed vs Battery (Colored by Charger)-Figure-25")
plt.show()

sns.kdeplot(df[df["Fast_Charger_Compatible"]==1]["Range_miles"], label="Compatible")
sns.kdeplot(df[df["Fast_Charger_Compatible"]==0]["Range_miles"], label="Not Compatible")
plt.title("Range Distribution by Charger Compatibility (KDE)-Figure-26")
plt.xlabel("Range (miles)")
plt.legend()
plt.show()

sns.kdeplot(df[df["Fast_Charger_Compatible"]==1]["Top_Speed_mph"], label="Compatible")
sns.kdeplot(df[df["Fast_Charger_Compatible"]==0]["Top_Speed_mph"], label="Not Compatible")
plt.title("Top Speed Distribution by Charger Compatibility (KDE)-Figure-27")
plt.xlabel("Top Speed (mph)")
plt.legend()
plt.show()

plt.scatter(df["Range_miles"], df["Battery_kWh"], s=df["Top_Speed_mph"]*2, c=df["Fast_Charger_Compatible"])
plt.xlabel("Range")
plt.ylabel("Battery")
plt.title("Range vs Battery (Size=Top Speed, Color=Charger)-Figure-28")
plt.show()

sns.stripplot(x="Brand", y="Battery_kWh", data=df, jitter=True)
plt.title("Battery Distribution by Brand-Figure-29")
plt.show()

sns.stripplot(x="Brand", y="Range_miles", data=df, jitter=True)
plt.title("Range Distribution by Brand-Figure-30")
plt.show()

brand_battery = df.groupby("Brand")["Battery_kWh"].mean().reset_index()

sns.heatmap(brand_battery.set_index("Brand").T, annot=True, cmap="YlGnBu")
plt.title("Average Battery by Brand Heatmap-Figure-31")
plt.show()