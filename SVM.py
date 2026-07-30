import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load Dataset 
iris = load_iris()
print("Feature names :", iris.feature_names)
print("Target names  :", iris.target_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"]  = iris.target
df["species"] = df["target"].apply(lambda x: iris.target_names[x])
print("\nDataFrame head:\n", df.head())

# Separate Classes for Plotting
setosa     = df[df.target == 0]
versicolor = df[df.target == 1]
virginica  = df[df.target == 2]

# Scatter Plot 1: Setosa vs Versicolor 
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(setosa["sepal length (cm)"],     setosa["sepal width (cm)"],
            color="b", marker="+", label="Setosa (sepal)")
ax1.scatter(versicolor["petal length (cm)"], versicolor["petal width (cm)"],
            color="r", marker=".", label="Versicolor (petal)")
ax1.set_xlabel("Length (cm)")
ax1.set_ylabel("Width (cm)")
ax1.set_title("Setosa Sepal vs Versicolor Petal")
ax1.legend()
fig1.tight_layout()
fig1.savefig(os.path.join(script_dir, "SVM-fig1.png"), dpi=150, bbox_inches="tight")
print("Saved: SVM-fig1.png")
plt.show()

#  Scatter Plot 2: Virginica vs Versicolor 
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.scatter(virginica["petal length (cm)"],  virginica["petal width (cm)"],
            color="g", marker=".", label="Virginica (petal)")
ax2.scatter(versicolor["sepal length (cm)"], versicolor["sepal width (cm)"],
            color="orange", marker="+", label="Versicolor (sepal)")
ax2.set_xlabel("Length (cm)")
ax2.set_ylabel("Width (cm)")
ax2.set_title("Virginica Petal vs Versicolor Sepal")
ax2.legend()
fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, "SVM-fig2.png"), dpi=150, bbox_inches="tight")
print("Saved: SVM-fig2.png")
plt.show()

#  Train / Test Split 
X = df.drop(columns=["target", "species"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Variants
models = {
    "Default SVC"           :SVC(),
    "C=8 (Regularised)"     :SVC(C=8),
    "Gamma=90"              :SVC(gamma=90),
    "RBF Kernel"            :SVC(kernel="rbf"),
}

print("\nSVM Accuracy Comparison")
print("-" * 35)
for label, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"{label:<25}: {accuracy:.2%}")
