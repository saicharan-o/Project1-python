def two_sum_optimized(nums, target):
    hashmap={}
    for i, num in enumerate(nums):
        diff=target-num
        if diff in hashmap:
            return [hashmap[diff],i]
        hashmap[num]=i


print(two_sum_optimized([2,0,10,15],10))   
print(two_sum_optimized([2,10,11,5],20))
print(two_sum_optimized([3,7,9,20],15)) 