import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir=os.path.dirname(os.path.abspath(__file__))

#Series
series=pd.Series([10,29,32,4,52,20])
print("Series:\n",series)

#DataFrame Construction & Inspection
data = {
    "Name":["Sai","Ram","Rolex","Vikram"],
    "Marks":[4,67,89,90],
    "Age":[34,6,89,84],
}

df=pd.DataFrame(data)
print("\nDataFrame:\n",df)
print("\nIndex:",df.index.tolist())
print("\nInfo:",df.info())
print("\nShape:",df.shape)
print("\nColumns:",df.columns.tolist())
print("\nCell [0, Age]:",df.loc[0,"Age"])
print("\nHead (3):\n",df.head(3))
print("\nTail (3):\n",df.tail(3))
print("\nDescribe:\n",df.describe())
print("\nRow 1:\n",df.loc[1])

#Fig 1: Marks per Student (Bar Chart)
fig1,ax1=plt.subplots(figsize=(7,4))
bars=ax1.bar(df["Name"],df["Marks"],color=["#4C72B0","#DD8452","#55A868","#C44E52"],
               edgecolor="white",linewidth=0.8)
ax1.bar_label(bars,padding=3,fontsize=9)
ax1.set_xlabel("Student")
ax1.set_ylabel("Marks")
ax1.set_title("Student Marks")
ax1.spines[["top","right"]].set_visible(False)
fig1.tight_layout()
fig1.savefig(os.path.join(script_dir,"Pandas_Basics-Fig-1.png"),dpi=150,bbox_inches="tight")
print("Saved: Pandas_Basics-Fig-1.png")
plt.show()

#Fig 2: Age per Student (Horizontal Bar Chart)
fig2,ax2=plt.subplots(figsize=(7,4))
ax2.barh(df["Name"],df["Age"],color=["#8172B3","#937860","#DA8BC3","#8C8C8C"],
         edgecolor="white",linewidth=0.8)
ax2.set_xlabel("Age")
ax2.set_ylabel("Student")
ax2.set_title("Student Ages")
ax2.spines[["top","right"]].set_visible(False)
fig2.tight_layout()
fig2.savefig(os.path.join(script_dir,"Pandas_Basics-Fig-2.png"),dpi=150,bbox_inches="tight")
print("Saved: Pandas_Basics-Fig-2.png")
plt.show()

#Missing Value Handling
data_with_nan = {
    "Name":  ["Sai","Ram",np.nan,"Vikram"],
    "Marks": [4,np.nan,89,90],
    "Age":   [34,6,89,np.nan],
}

df_nan=pd.DataFrame(data_with_nan)
print("\nDataFrame with NaN:\n",df_nan)
print("\nNull mask:\n",df_nan.isnull())
print("\nFill NaN with 2:\n",df_nan.fillna(2))
print("\nDrop NaN rows:\n",df_nan.dropna())

#Fig 3: Null Value Heatmap
fig3,ax3=plt.subplots(figsize=(5,4))
null_mask = df_nan.isnull().astype(int)
im = ax3.imshow(null_mask, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax3.set_xticks(range(len(df_nan.columns)))
ax3.set_xticklabels(df_nan.columns)
ax3.set_yticks(range(len(df_nan)))
ax3.set_yticklabels([f"Row {i}" for i in df_nan.index])
ax3.set_title("Missing Value Map\n(Red = NaN, Green = Present)")
plt.colorbar(im,ax=ax3,ticks=[0,1],label="Is Null")
fig3.tight_layout()
fig3.savefig(os.path.join(script_dir,"Pandas_Basics-Fig-3.png"),dpi=150,bbox_inches="tight")
print("Saved: Pandas_Basics-Fig-3.png")
plt.show()

#Filtering & Derived Column
print("\nMarks > 20:\n",df_nan[df_nan["Marks"]>20])
print("\nMarks > 15 & Age > 20:\n",df_nan[(df_nan["Marks"]>15) & (df_nan["Age"]>20)])

df_nan["Passed"] = df_nan["Marks"] > 13
print("\nWith 'Passed' column:\n",df_nan)
print("\nDropped 'Passed':\n",df_nan.drop("Passed",axis=1))
print("\nRenamed 'Marks' to 'Score':\n",df_nan.rename(columns={"Marks": "Score"}))

#Fig 4: Marks sorted Ascending
df_sorted = df.sort_values(by="Marks", ascending=True).reset_index(drop=True)
fig4, ax4 = plt.subplots(figsize=(7, 4))
ax4.plot(df_sorted["Name"], df_sorted["Marks"], marker="o",
         color="#4C72B0", linewidth=2, markersize=8, markerfacecolor="#DD8452")
for i, (name, mark) in enumerate(zip(df_sorted["Name"], df_sorted["Marks"])):
    ax4.annotate(str(mark), (name, mark), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9)
ax4.set_xlabel("Student (sorted by Marks)")
ax4.set_ylabel("Marks")
ax4.set_title("Marks - Ascending Order")
ax4.spines[["top", "right"]].set_visible(False)
fig4.tight_layout()
fig4.savefig(os.path.join(script_dir, "Pandas_Basics-Fig-4.png"), dpi=150, bbox_inches="tight")
print("Saved: Pandas_Basics-Fig-4.png")
plt.show()

print("\nGroupBy Age -> mean Marks:\n",df_nan.groupby("Age")["Marks"].mean())
