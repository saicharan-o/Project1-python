import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def move_zeroes(nums: list[int]) -> None:
    """
    Move all zeroes to the end of `nums` in-place while maintaining
    the relative order of non-zero elements.

    Args:
        nums: List of integers to rearrange in-place.
    """
    insert_pos = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[insert_pos], nums[i] = nums[i], nums[insert_pos]
            insert_pos += 1


# Test Cases
test_cases=[
    [2,3,0,20,0,0,1,1,2,2,0],
    [0,3,4,5,0,1,0,0,2,3,4,0],
    [0,0,1,0,0,2,0],
    [2,3,9,2,4,5],
    [0,30,40,5,0,1,0,0,0],
    [0,0,33,0,92,0],
    [2,3,99,0,9,0,9,0,1,0,2,0,3,0,4,0],
    [0,3,4,5,0,0,2,3,6,7,0,0,8,9,0],
    [0,0,0,0,1,2,3,0,0,0,4,5,6,0,0,0,7,8,9,0,0,0],
]

print("Move Zeroes Results")
print("="*50)

# Store before/after for visualisation
originals = [lst.copy() for lst in test_cases]
for idx, nums in enumerate(test_cases, start=1):
    move_zeroes(nums)
    print(f"Test {idx:>2}: {nums}")

# Visualisation: Before vs After 
fig, axes = plt.subplots(3,3,figsize=(15,10))
fig.suptitle("Move Zeroes — Before vs After",fontsize=16,fontweight="bold",y=1.01)
fig.patch.set_facecolor("#1E1E2E")

ZERO_COLOR="#EF5350"   # red for zeros
NONZERO_BEFORE="#90CAF9"  # blue for non-zeros (before)
NONZERO_AFTER="#66BB6A"  # green for non-zeros (after)

for ax,before,after,idx in zip(
    axes.flatten(),originals,test_cases,range(1, 10)
):
    x = np.arange(len(before))

    # Before bars
    colors_before = [ZERO_COLOR if v == 0 else NONZERO_BEFORE for v in before]
    ax.bar(x - 0.2, before, width=0.35, color=colors_before, alpha=0.85, label="Before")

    # After bars
    colors_after = [ZERO_COLOR if v == 0 else NONZERO_AFTER for v in after]
    ax.bar(x + 0.2, after,  width=0.35, color=colors_after,  alpha=0.85, label="After")

    ax.set_title(f"Test {idx}", color="white", fontsize=10)
    ax.set_facecolor("#2D2D44")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")

legend_elements = [
    mpatches.Patch(color=NONZERO_BEFORE, label="Non-zero (Before)"),
    mpatches.Patch(color=NONZERO_AFTER,  label="Non-zero (After)"),
    mpatches.Patch(color=ZERO_COLOR,     label="Zero"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           facecolor="#1E1E2E", labelcolor="white", fontsize=10,
           bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()

script_dir  = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "Zero-Figure.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nSaved: {output_path}")
plt.show()