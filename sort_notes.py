""" 
Sorting algorithms
"""

from typing import List 

# Bubble sort 
def bubble_sort(nums: List[int]) -> List[int]:
    for _ in range(len(nums)-1):
        for i in range(len(nums) - 1 ):
            if nums[i] > nums[i+1]:
                nums[i], nums[i+1] = nums[i+1], nums[i]
    return nums

nums = [3,2,4,1,5,6]
print(bubble_sort(nums))
# [1, 2, 3, 4, 5, 6]
# T: O(n^2)


# Insertion sort 
def insertion_sort(nums: List[int]) -> List[int]:
    for i in range(len(nums)):    
        cur = nums.pop(i)    # pop current item
        j = i -1             # pointer to prev item
        while j >=0 and nums[j] > cur:     # if cur < prev, keep move to left small values
            j -= 1 
        nums.insert(j+1, cur)    # insert cur when cur > prev is found 

    return nums 

nums = [3,2,4,1,5,6]
print(insertion_sort(nums))
# [1, 2, 3, 4, 5, 6]


# Merge sort, divide and conquer
# split arry into half, till single element array 
# sort single element array
# merge back together 

# helper function to merge left and right list 
def merge(left, right):
    result = [] 
    while True:
        if len(left) > 0:
            if len(right) > 0:
                if left[0] <= right[0]:
                    result.append(left[0])
                    left.pop(0)
                else:
                    result.append(right[0])
                    right.pop(0)
            else:
                result.append(left[0])
                left.pop(0)
        elif len(right) > 0:
            result.append(right[0])
            right.pop(0)
        else: 
            break
    return result 

# merge optimized, keeps track 2 indices going through 2 lists 
# to boost performance 
def merge_optimized(left, right):
    result = [] 
    left_ind, right_ind = 0, 0 
    left_len, right_len = len(left), len(right)
    while True:
        if left_ind < left_len:
            left_val = left[left_ind]
            if right_ind < right_len:
                right_val = right[right_ind]
                if left_val <= right_val:
                    result.append(left_val)
                    left_ind += 1 
                else:
                    result.append(right_val)
                    right_ind += 1  
            else:
                result.append(left_val)
                left_ind += 1 
        elif right_ind < right_len:
            result.append(right[right_ind])
            right_ind += 1 
        else:
            break 
    return result 

# merge_sort without variable 
def merge_sort_no_var(l):
    if len(l)<=1: 
        return l 
    return merge(    # no any var 
            merge_sort_no_var(l[:int(len(l)/2)]),
            merge_sort_no_var(l[int(len(l)/2):]))

def merge_sort_with_var(l):
    if len(l) <= 1:
        return l
    mid = len(l)//2    # using var mid 
    # return merge(
    return merge_optimized (
            merge_sort_with_var(l[:mid]),
            merge_sort_with_var(l[mid:]))

def merge_sort_1(nums: List[int]) -> List[int]:
    if len(nums) > 1:
        mid = len(nums) // 2 
        l = nums[:mid]
        r = nums[mid:]
        merge_sort_1(l)
        merge_sort_1(r)

        i = j = k = 0 

        while i < len(l) and j < len(r):
            if l[i] <= r[j]:
                nums[k] = l[i]
                i += 1 
            else:
                nums[k] = r[j]
                j += 1 
            k += 1 
        
        while i < len(l):
            nums[k] = l[i]
            i += 1
            j += 1 
        
        while j < len(r):
            nums[k] = r[j]
            j += 1 
            k += 1 

    return nums

nums = [3,2,4,1,7,5,6]
print(f"{nums=}")
print(merge_sort_no_var(nums)) 
print(merge_sort_with_var(nums))

# print(merge_sort_0(nums))  
# nums=[3, 2, 4, 1, 5, 6]
# [1, 2, 3, 4, 5, 6]


# Quick sort, sort in place
# T: O(nlogn)