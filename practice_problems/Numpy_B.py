import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

script_dir=os.path.dirname(os.path.abspath(__file__))

#Performance: Python List vs NumPy Array
SIZE=100_000
l1,l2=range(SIZE),range(SIZE)
start=time.time()
_=[(x+y)for x,y in zip(l1,l2)]
list_time=(time.time()-start)*1000
print(f"Python list addition : {list_time:.4f} ms")
a1,a2=np.arange(SIZE),np.arange(SIZE)
start=time.time()
_=a1+a2
numpy_time=(time.time()-start)*1000
print(f"NumPy array addition : {numpy_time:.4f} ms")

#Fig 1: Performance Comparison Bar Chart
fig1,ax1=plt.subplots(figsize=(6,4))
bars=ax1.bar(["Python List","NumPy Array"],[list_time,numpy_time],
               color=["#EF5350","#42A5F5"],edgecolor="white", linewidth=0.8, width=0.5)
ax1.bar_label(bars,fmt="%.3f ms",padding=4,fontsize=9)
ax1.set_ylabel("Time (ms)")
ax1.set_title(f"Performance: Python List vs NumPy\n(N = {SIZE:,} elements)")
ax1.spines[["top","right"]].set_visible(False)
fig1.tight_layout()
fig1.savefig(os.path.join(script_dir,"Numpy_Basics-fig-1.png"),dpi=150, bbox_inches="tight")
print("Saved: Numpy_Basics-fig-1.png")
plt.show()

#3-D Array Inspection
arr=np.array([[[1,2,3],[7,4,8]],[[13,4,68],[4,5,7]]])
print("\nArray:\n",arr)
print("Dimensions:",arr.ndim)
print("Shape:",arr.shape)
print("Size:",arr.size)
print("Data type:",arr.dtype)

#Array Creation Utilities
print("\nZeros:\n",np.zeros((2,3)))
print("Ones:\n",np.ones((2,2)))
print("Full (8s):\n",np.full((4,3),8))
print("Identity matrix:\n",np.eye(6))
print("arange(1,10,2):",np.arange(1,10,2))
print("linspace(1,20,6):",np.linspace(1,20,6))

#Element-wise Operations
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print("\nElement-wise multiplication:",a*b)
print("Element-wise addition:",a+b)
print("Element-wise subtraction:",a-b)
print("Element-wise modulo:",a%b)
print("Element-wise division:",a/b)
print("Scalar × 2:",a*2)
print("Scalar × 3:",b*3)

#Statistical Operations
data=np.array([24,67,23,82,79,24,90])

print("\nArray:",data)
print("Sum:",np.sum(data))
print("Max:",np.max(data))
print("Min:",np.min(data))
print("Mean:",np.mean(data))
print("Std:",np.std(data))

matrix=np.array([[1,3,6,8],[42,6,8,2]])
print("\nColumn-wise sum:",np.sum(matrix,axis=0))
print("Row-wise sum:",np.sum(matrix,axis=1))

#Fig 2: Statistics Bar Chart
stats={
    "Sum":float(np.sum(data)),
    "Max":float(np.max(data)),
    "Min":float(np.min(data)),
    "Mean":float(np.mean(data)),
    "Std":float(np.std(data)),
}
fig2,axes2=plt.subplots(1,2,figsize=(11,4))

#Left: raw data bar chart
axes2[0].bar(range(len(data)),data,color="#5C6BC0",edgecolor="white",linewidth=0.8)
axes2[0].set_xticks(range(len(data)))
axes2[0].set_xticklabels([str(v) for v in data])
axes2[0].set_title("Raw Data Array")
axes2[0].set_ylabel("Value")
axes2[0].axhline(float(np.mean(data)), color="#EF5350", linestyle="--",
                  linewidth=1.5, label=f"Mean = {np.mean(data):.1f}")
axes2[0].legend(fontsize=9)
axes2[0].spines[["top", "right"]].set_visible(False)

#Right: statistics summary
stat_names=["Sum","Max","Min","Mean","Std"]
stat_values=[stats[k] for k in stat_names]
colors2=["#42A5F5","#66BB6A","#EF5350","#FFA726","#AB47BC"]
b2=axes2[1].bar(stat_names,stat_values,color=colors2,edgecolor="white",linewidth=0.8)
axes2[1].bar_label(b2,fmt="%.1f",padding=3,fontsize=9)
axes2[1].set_title("Statistical Summary")
axes2[1].set_ylabel("Value")
axes2[1].spines[["top", "right"]].set_visible(False)

fig2.suptitle("NumPy Statistical Operations",fontweight="bold")
fig2.tight_layout()
fig2.savefig(os.path.join(script_dir,"Numpy_Basics-fig-2.png"),dpi=150,bbox_inches="tight")
print("Saved: Numpy_Basics-fig-2.png")
plt.show()

#Broadcasting
mat=np.array([[1,3,6,8],[42,6,8,2]])
vec=np.array([10,40,60,80])

print("\nBroadcast addition:",mat+vec)
print("Broadcast multiplication:",mat*vec)
print("Broadcast subtraction:",mat-vec)
print("Broadcast modulo:",mat%vec)
print("Broadcast division:",mat/vec)
print("Broadcast scalar × 2:",mat*2)
print("Broadcast scalar × 3:",vec*3)

#Copy vs View
original=np.array([2,9,4,5])
copy=original.copy()
copy[2]=304
print("\nOriginal array:",original)
print("Modified copy:",copy)

#Random Number Generation
rand_floats=np.random.rand(3,3)
rand_ints=np.random.randint(1,10,(3,3))
print("\nRandom floats (3×3):\n",rand_floats)
print("Random integers (3×3):\n",rand_ints)

#Sorting & Boolean Indexing
unsorted=np.array([8,492,4813,4,775,1,9])
sorted_arr=np.sort(unsorted)
print("\nSorted array:",sorted_arr)

filtered_arr=np.array([23,494,39,3,28,53,40])
above_35=filtered_arr[filtered_arr>35]
print("Values > 35:",above_35)

#Fig 3: Sorting & Filtering Side-by-Side
fig3,axes3=plt.subplots(1,2,figsize=(11,4))

#Left: before/after sorting
x_labels=[str(v)for v in unsorted]
axes3[0].bar(x_labels,unsorted,color="#90CAF9",edgecolor="white",label="Unsorted",alpha=0.8)
axes3[0].plot(x_labels,sorted_arr,color="#EF5350",marker="o",linewidth=2,markersize=6,label="Sorted order")
axes3[0].set_title("Array Sorting")
axes3[0].set_ylabel("Value")
axes3[0].set_xlabel("Original positions")
axes3[0].legend(fontsize=9)
axes3[0].spines[["top", "right"]].set_visible(False)

#Right: boolean indexing highlight
f_labels=[str(v)for v in filtered_arr]
bar_colors=["#66BB6A" if v>35 else "#EF5350" for v in filtered_arr]
b3=axes3[1].bar(f_labels,filtered_arr,color=bar_colors,edgecolor="white",linewidth=0.8)
axes3[1].axhline(35,color="#FFA726",linestyle="--",linewidth=1.5,label="Threshold = 35")
axes3[1].set_title("Boolean Indexing (Values > 35)")
axes3[1].set_ylabel("Value")
axes3[1].set_xlabel("Element value")
axes3[1].legend(fontsize=9)
axes3[1].spines[["top","right"]].set_visible(False)

import matplotlib.patches as mpatches
green_patch=mpatches.Patch(color="#66BB6A",label="Passes filter (>35)")
red_patch=mpatches.Patch(color="#EF5350",label="Filtered out (<=35)")
axes3[1].legend(handles=[green_patch,red_patch],fontsize=9)
fig3.suptitle("NumPy Sorting & Boolean Indexing",fontweight="bold")
fig3.tight_layout()
fig3.savefig(os.path.join(script_dir,"Numpy_Basics-fig-3.png"),dpi=150,bbox_inches="tight")
print("Saved: Numpy_Basics-fig-3.png")
plt.show()
