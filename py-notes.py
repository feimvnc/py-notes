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