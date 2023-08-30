""" 
keep some useful python snippets for references

# python document download
https://docs.python.org/dev/download.html
"""


# random integer generation k=10 
from random import choices
arr = choices(range(10), k=10)       # when use [choices], it will be [[2,2,...]]
print(f"{arr=}")
# arr=[2, 2, 4, 1, 8, 4, 8, 9, 1, 7]


# find the max sub-string with no repetative chars 
def longest_unique_substr(s: str) -> int:
    longest = 0 
    for left in range(len(s)):         # starts from 0 and move to next 
        contains=set()                 # create a new set for each start_index 
        for char in s[left:]:          # slice from start_index till end 
            if char in contains:       # duplicate found 
                break
            contains.add(char)         # not executed when break 
        # execute only after break 
        longest = max(longest, len(contains))    # max of prev and new length, after break
    return longest 

s1='abcabcbb'
s2='bbbbb'
print(f"{s1=}:", longest_unique_substr(s1))
print(f"{s2=}:", longest_unique_substr(s2))
# s1='abcabcbb': 3
# s2='bbbbb': 1


# dict initializations 
code = [('usa', 1),('uk', 44),('india',91)]    
d = {k: v for (k, v) in code}    # dict comprehension
print(f'{d=}') 
# d={'usa': 1, 'uk': 44, 'india': 91}

lst = [('a', 2), ('b', 4), ('c', 6)]    # from tuple of two values list 
print(f"{dict(lst)=}")
# dict(lst)={'a': 2, 'b': 4, 'c': 6}

list1, list2 = ['a', 'b', 'c'], [1,2,3]
d = dict(zip(list1, list2))
print(f'{d=}')
# d={'a': 1, 'b': 2, 'c': 3}


# regex word counts 
import re
words=re.findall('\w+', open('/etc/hosts').read().lower())
print(f"{len(words)=}")
# len(words)=38


# deque 
from collections import deque 
def tail(filename, n=10):
    'Return the last n lines of a file'
    with open(filename) as f:
        return deque(f, n)
print("".join(tail('/etc/hosts', 2)))
# ::1             localhost
# 192.168.10.2    docker.local


import itertools
def moving_average(iterable, n=3):
    # moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    # https://en.wikipedia.org/wiki/Moving_average
    it = iter(iterable)
    d = deque(itertools.islice(it, n-1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n

print(list(moving_average([40, 30, 50, 46, 39, 44])))
# [40.0, 42.0, 45.0, 43.0]


# del d[n] using rotate() to position elements to be popped
def delete_nth(d, n):
    d.rotate(-n)
    d.popleft()
    d.rotate(n)

d = deque([1,2,3,4,5])
print(f'before {d=}')
delete_nth(d, 2)
print(f'after {d=}')
# before d=deque([1, 2, 3, 4, 5])
# after d=deque([1, 2, 4, 5])


# defaultdict counting like a bag or multiset
from collections import defaultdict
s = 'mississippi'
d = defaultdict(int)
for k in s:
    d[k] += 1

print(f"{sorted(d.items())=}")
# sorted(d.items())=[('i', 4), ('m', 1), ('p', 2), ('s', 4)]


# ordeal (ord) number for alphabetical letters 
import string 
d = {k: ord(k) for (k, v) in zip(string.ascii_lowercase, range(26))}
print(f'{d=}')
# d={'a': 97, 'b': 98, 'c': 99, 'd': 100, 'e': 101, 'f': 102, 'g': 103, 'h': 104, 
# 'i': 105, 'j': 106, 'k': 107, 'l': 108, 'm': 109, 'n': 110, 'o': 111, 'p': 112, 
# 'q': 113, 'r': 114, 's': 115, 't': 116, 'u': 117, 'v': 118, 'w': 119, 'x': 120, 
# 'y': 121, 'z': 122}


# heapq heapify heapsort operations
import heapq
nums = [6,4,2,1,5,3]
print(f'before heapify {nums=}')
heapq.heapify(nums) # Heap queue is minheap, min element at the top of the queue
print(f'after heapify {nums=}')
result = []
while nums:
    result.append(heapq.heappop(nums))
print(f'after heap sort {result=}')


# queue by list, FIFO
queue = []    # create a queue 
for i in range(10):
    queue.append(i)    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
while queue:
    print(queue.pop(0), end = ' ')    # 0 1 2 3 4 5 6 7 8 9 


# stack by list, LIFO 
stack = [] 
for i in range(10):
    stack.append(i)    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
while stack: 
    print(stack.pop(), end = ' ')    # 9 8 7 6 5 4 3 2 1 0 


# heapsort
from heapq import heappush, heappop 
def heapsort(iterable):
    h = [] 
    for value in iterable: 
        heappush(h, value)
    return [heappop(h) for i in range(len(h))]

print("\n", heapsort([2,3,1,5]))
# [1, 2, 3, 5]


# bfs graph traversal, queue, list 
def bfs(graph, node):
    visited = [] 
    queue = [] 

    visited.append(node)
    queue.append(node)

    while queue:
        s = queue.pop(0)        # pop the item
        print(s, end = ' ')

        for n in graph[s]:
            if n not in visited:
                visited.append(n)   # append the item 
                queue.append(n)

# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': [''],
#     'D': [''],
#     'E': ['']
# }


# dfs graph traversal, stack, list 


# recursive tree walk
def walk(tree):
    if tree is not None:
        print(tree)
        walk(tree.left)
        walk(tree.right)


# dfs, stack tree walk
class Node: 
    def __init__(self, value, left=None, right=None):
        self.value = value 
        self.left = left 
        self.right = right 
    def __str__(self):
        return f"Node {str(self.value)=}"

def walk_recursive(tree):
    if tree is not None:
        print(tree)
        walk(tree.left)
        walk(tree.right)

def walk_stack(tree, stack):
    stack.append(tree)
    while len(stack) > 0:
        node = stack.pop()
        if node is not None:
            print(node)
            stack.append(node.right)
            stack.append(node.left)
        

mytree = Node('A', Node('B', Node('D'), Node('E')),
            Node('C', Node('F'), Node('G')))
walk_recursive(mytree)
s = []
walk_stack(mytree, s)

# Node str(self.value)='B'
# Node str(self.value)='D'
# Node str(self.value)='E'
# Node str(self.value)='C'
# Node str(self.value)='F'
# Node str(self.value)='G'
# Node str(self.value)='A'
#
# Node str(self.value)='B'
# Node str(self.value)='D'
# Node str(self.value)='E'
# Node str(self.value)='C'
# Node str(self.value)='F'
# Node str(self.value)='G'


# Five common graph algorithms 
""" 
Depth First Search (DFS), Time: O(n), Data Structures: HashSet, Stack
Breadth First Search (BFS), Time: O(n), Data Structures: Queue, HashSet
Union Find, Time: O(nlogn), Data Structure: Forest of Trees
Topological Sort: Time: O(n), Data Structures: HashSet
Dijkstra's Shortest Path: Time: ElogV (E: number of edges, V: vertices), Data Structures: Heap, HashSet
"""


# binary search target k in 1D array
def binary_search(nums, k) -> bool:
    l, r = 0, len(nums) - 1
    while l <= r:    # must include equal condition, when l==r and k is found 
        m = ( l + r ) // 2 
        if k > nums[m]:
            l = m + 1
        elif k < nums[m]:
            r = m - 1 
        else:    # k = nums[m] condition 
            return True
    return False

    # while left < right:   # for [2,2] condition
    #     # ...
    # return left if nums[left] == target else -1

nums = [ i for i in range(10)]
k = 3
print(binary_search(nums, k))
# True


# maximum sum of contiguous subarray, kadane algorithm 
from typing import List 
def kadane(a: List[int]) -> int:
    meh = msf = a[0]    # meh: max elem here, msf: max elem so far
    for i in range(1, len(a)):    # loop through array, from 1 till end, 0 already used above 
        meh = meh + a[i]     # update meh 
        if meh < a[i]:       
            meh = a[i]       # update meh 
        if msf < meh:
            msf = meh        # update msf, new max value found 
    return msf

a=[-2,-3,4,-1,-2,1,5,-3]
print(kadane(a))
# 7


# fast-inverse-square-root, 0x5f3759df, from famous Quake 3 game 
# https://github.com/ajcr/ajcr.github.io/blob/master/_posts/2016-04-01-fast-inverse-square-root-python.md
# just for fun, implement in python 
import numpy as np 
def numpy_isqrt(number):
    threehalfs = 1.5
    x2 = number * 0.5
    y = np.float32(number)
    
    i = y.view(np.int32)                            # float to int 
    i = np.int32(0x5f3759df) - np.int32(i >> 1)     # int arithmetic
    y = i.view(np.float32)                          # back to float 
    
    y = y * (threehalfs - (x2 * y * y))             # newton's method  
    return float(y)

# python built-in is fast and accurate, this is different from C, always use built-in
def builtin_isqrt(number):
    return number ** -0.5

# check execution time 
import timeit
print(f"numpy_isqrt: {timeit.timeit('numpy_isqrt(22)', number=10000, globals=globals())}")   # without globals, error not defined
print(f"builtin_isqrt: {timeit.timeit('builtin_isqrt(22)', number=10000, globals=globals())}")   
# numpy_isqrt: 0.09822194697335362
# builtin_isqrt: 0.0015821419656276703     # much faster 


# print array from last index len(freq)-1, up to 0 excluding -1, in descending order -1 
nums = [n for n in range(5)]    # [0, 1, 2, 3, 4]
for index in range(len(nums)-1, -1, -1):
    print(f'{index=}, {nums[index]=}')
# index=4, nums[i]=0
# index=3, nums[i]=0
# index=2, nums[i]=0
# index=1, nums[i]=0
# index=0, nums[i]=0


# python vs C, reverseLeftWords
# python 
def reverseLeftWords(s: str, n: int) -> str:
    return s[n:] + s[:n]

# C 
''' 
char *reverseSelectWords(char *s, int n)
{
    int len = strlen(s), k=0;
    char *res = malloc((len+1) * sizeof(char));
    char *p = s + n;
    for (int i  = 0; i < len - n, i++)
    {
        res[k++] = p[i]
    }
    for (int i=0; i  < n; i++)
    {
        res[k++] = s[i];
    }
    res[k] = '\0';
    return res;
}
'''


# recursion: an approach to solving problems using a function that calls itself as a subroutine.
# most common usages: memorization, divide and conquer, backtracking
# def recursion(arguments):
#     base case:    # termination 
#     recursion_relation    # divide big into small 
""" 
cache = dict()    # initialize cache (same as a dict)

def foo(arg):
    if arg in cache:
        return cach[arg]
    # base condition
    # recursion relation
    cache[arg] = result 
"""
# fibonacci number 
# no memorization 
def fib_recur(n):
    if n<2: 
        return n
    else: 
        return fib_recur(n-1)+fib_recur(n-2)

# with memorization
cache = dict()
def fib_cache(n):
    if n in cache: 
        return cache[n]
    if n < 2:
        result = n 
    else: 
        result = fib_cache(n-1) + fib_cache(n-2)
    cache[n] = result 
    return result 


## string comparison  
def strcmp(s1, s2):
    i1, i2 =0, 0
    while i1 < len(s1) and i2 < len(s2):
        if ord(s1[i1]) == ord(s2[i2]):
            i1 += 1
            i2 += 1
        elif ord(s1[i1]) < ord(s2[i2]):
            return -1
        else:
            return 1 
        
        if len(s1) < len(s2):
            return -1
        elif len(s1) > len(s2):
            return 1
        else:
            return 0

print(strcmp('hello', 'hello'))
print(strcmp('hello', 'world'))
# 0
# -1


# Type hinting 
from typing import Sequence, Tuple, Any 
# Sequence: simpliest abstract base class that supports __getitem__ syntax
# Any: generic type 
def  get_first_and_last(x: Sequence[Any]) -> Tuple[Any, Any]:
    return (x[0], x[-1])

print(get_first_and_last([1,2,3,4,5]))
# (1, 5)


## map for broadcast operation
print(list(map(str.upper, 'hello')))
print(list(map(pow, [1,2,3,4,5],[2,2,2,2,2])))    # requires two lists for pow
# ['H', 'E', 'L', 'L', 'O']
# [1, 4, 9, 16, 25]

def transform_double(base: int) -> int:
    return base * base
print(list(map(transform_double, [1,2,3,4,5])))
# [1, 4, 9, 16, 25]

def is_iterable(iterable: Any) -> bool:  
    try:
        iter(iterable)
        return True 
    except TypeError:
        return False 

print(list(map(is_iterable, [1, [2,3], 3.14, 'hello'])))
# [False, True, False, True]


# functools cache
from functools import cache, lru_cache 
@cache     # return the same and faster than lru_cache
def fac(n: int) -> int:
    return n * fac(n-1) if n else 1

@lru_cache(maxsize=None)
def count_vowels(s: str) -> int:
    return sum(s.count(vowel) for vowel in 'AEIOUaeioe')

print(fac(10))
print(count_vowels("mississippi"))
print(fac.cache_info())
print(count_vowels.cache_info())
# 3628800
# 4
# CacheInfo(hits=0, misses=11, maxsize=None, currsize=11)
# CacheInfo(hits=0, misses=1, maxsize=128, currsize=1)


# partial is used in function which freezes portion of argument is freezed
from functools import partial 
bin2int = partial(int, base=2)
hex2int = partial(int, base=16)
bin2int.__doc__ = "Convert base 2 string to an int"
hex2int.__doc__ = "Convert base 16 string to an int"
print(bin2int('1010'))
print(hex2int('10'))
# 10
# 16


# reduce apply function of two arguments cumulativelly to items of iterables 
from functools import reduce 
# equivalent to ((((1+2)+3)4)+5), + can be -, *, /
print(reduce(lambda x, y: x + y, [1,2,3,4,5]))
# 15


# Path write and read a file 
from pathlib import Path 
# write a file
p = Path('test.txt')
p.write_text('hello world path')
print(p.exists()) 
# read a file 
q = Path('test.txt')
with p.open() as f: print(f.readline())
# True
# hello world path


# tempfile create secure file in memory 
import tempfile 
fp = tempfile.TemporaryFile()
fp.write(b'hello world')
fp.seek(0)
print(fp.read())
fp.close()     # file is deleted and removed from meory
# b'hello world'


# pickle python object serialization, pickle is not secure, only unpickle data you trust
import pickle 
data = [i for i in range(10)]
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

with open('data.pickle',  'rb') as f:
    data = pickle.load(f)
    print(data)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


