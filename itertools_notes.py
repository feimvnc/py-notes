""" 
Itertools offer high performance as the underlying toolset.
Superior memory performance is kept by processing elements one at a time 
    rather than bringing the whole iterable into memory all at once.
Code volume is kept small by linking the tools together in a functional style 
    which helps eliminate  temporary variable.
High speed is retained by preferring "vectorized" building blocks over 
    the use of for-loops and generators which incur interpreter overhead.

Below codes has additional notes and test cases 

source: https://docs.python.org/3/library/itertools.html
"""

# code 
import collections
import functools 
import math 
import operator 
import random 
import itertools

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))

def chain(*iterables):
    # chain('abc', 'def') -> a b c d e f
    for it in iterables:
        for element in it:
            yield element

def prepend(value, iterator):
    "Prepend a single value in front of an iterator"
    # prepend(1, [1,2,3]) -> 1 2 3 4 
    return chain([value], iterator)

def count(start=0, step=1):
    # count(10) --> 10 11 12 13 14 ...
    # count(2.5, 0.5)  --> 2.5 3.0 3.5 ...
    n = start 
    while True:
        yield n 
        n += step 

# func should be a function that accepts one integer argument.
# If start is not specified it defaults to 0. 
# It will be incremented each time the iterator is advanced.
def tabulate(function, start=0):
    "Return function(0), function(1), ..."
    return map(function, count(start))

def tail(n, iterable):
    " Return an iterator over the last n items"
    # tail (3, 'abcdefg') -> e f g 
    return iter(collections.deque(iterable, maxlen=n))

def consume(iterator, n=None):
    "Advance the iterator n-steps ahead.  If n is None, consume entirely."
    # Use functions that consume iterators at C speed  
    if n is None:
        # feed the entire iterator into a zero-length deque 
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice  starting at position n 
        next(itertools.islice(iterator, n, n), None)

def nth(iterable, n, default=None):
    "Return the nth item or a default value"
    return next(itertools.islice(iterable, n, None), default)    # stop, start (None)

def all_equal(iterable):
    "Return TTrue if all the elements are equal to each other"
    # all_equal('aaaab'))   -> all_equal False
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

def quantity(iterable, pred=bool):
    "Count how many times the predicate is True"
    # list(map(bool, 'abc)) -> [True, True, True]
    # quantity 4
    return sum(map(pred, iterable))

def ncycles(iterable, n):
    "Return the sequence elements n times"
    # ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter"
    # batched('abcdefg', 3) -> abc def g 
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch 

def grouper(iterable, n, *, incomplete='fill', fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('abcdefg', 3, fillvalue='x') -> abc def gxx
    # grouper('abcdefg', 3, incomplete='strict') -> abc def ValueError 
    # grouper('abcdefg', 3, incomplete='ignore') -> abc def 
    # it = [iter('abcdefg')] 
    # list(zip(*it)) -> [('a',), ('b',), ('c',), ('d',), ('e',), ('f',), ('g',)]
    # it = [iter('abcdefg')] * 2
    # list(zip(*it))  -> [('a', 'b'), ('c', 'd'), ('e', 'f')]
    # list(grouper('abcdefg', 3)) -> [('a', 'b', 'c'), ('d', 'e', 'f'), ('g', None, None)]
    args = [iter(iterable)] * n 
    if incomplete == 'fill':
        return itertools.zip_longest(*args, fillvalue=fillvalue)
    if incomplete == 'strict':
        return zip(*args, strict=True)
    if incomplete == 'ignore':
        return zip(*args)
    else:
        return ValueError('Expected fill, strict, or ignore')

def sumprod(vec1, vec2):
    "Compute a sum of products"
    # v1 = [1,2]
    # v2 = [3,4]
    # list(zip(v1, v2))  -> [(1, 3), (2, 4)]
    # list(itertools.starmap(operator.mul, zip(v1, v2)))  -> [3, 8]
    # return sum(itertools.starmap(operator.mul, zip(vec1, vec2, strict=True)))
    return sum(itertools.starmap(operator.mul, zip(vec1, vec2)))    # -> 11 = (3+8)

def sum_of_squares(it):
    "Add up the squares of the input values"
    # sum_of_squares([1,2,3]) -> 1 + 4 + 9 = 14
    # it = itertools.tee([1,2,3])
    # [list(c) for c in it]    ->  [[1, 2, 3], [1, 2, 3]]
    return sumprod(*itertools.tee(it))

def transpose(it):
    "Swap the rows and columns of the input"
    # transpose([(1,2,3),(11,22,33)]) -> (1,11)(2,22)(3,33)
    return zip(*it)

def matmul(m1, m2):
    "Multiply two metrics"
    # matmul([(7,5),(3,5)], [[2,5][7,9]]) -> (49,80),(41,60)
    # list(matmul([(1,2), (3, 4)], [[5, 6], [7, 8]])) -> [(19, 22), (43, 50)]
    n = len(m2[0])
    return batched(itertools.starmap(sumprod, itertools.product(m1, transpose(m2))), n)

def convolve(signal, kernel):
    # See:  https://betterexplained.com/articles/intuitive-convolution/
    # convolve(data, [0.25, 0.25, 0.25, 0.25]) --> Moving average (blur)
    # convolve(data, [1, -1]) --> 1st finite difference (1st derivative)
    # convolve(data, [1, -2, 1]) --> 2nd finite difference (2nd derivative)
    # list(convolve([1,1,1,1], [0.25, 0.25, 0.25, 0.25])) -> [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
    kernel = tuple(kernel)[::-1]   # reverse 
    n = len(kernel)
    # collections.deque([0], maxlen=5)*5  -> deque([0, 0, 0, 0, 0], maxlen=5)
    window = collections.deque([0], maxlen=n) * n 
    for x in chain(signal, itertools.repeat(0, n-1)):
        window.append(x)
        yield sumprod(kernel, window)

def polynomial_from_roots(roots):
    """ 
    Compute a polynomial's coefficient from its roots.
    (x-5)(x+4)(x-3) expands to: x^3 -4x^2 -17x + 60 
    """
    # polynomial_from_roots([5, -4, 3]) -> [1, -4, -17, 60]
    expansion = [1]
    for r in roots:
        expansion = convolve(expansion, (1, -r))
    return list(expansion)

def polynomial_eval(coefficients, x):
    """ 
    Evaluate a polynomial at a specific value.
    Computes with better numeric stability than Horner's method.
    """
    # Evaluate x^3 -4x^2 -17x +60 at x = 2.5  
    # polynomial_eval([1, -4, -17, 60], x=2.5) -> 8.125 
    n = len(coefficients)
    if n ==0:
        return x * 0 
    powers = map(pow, itertools.repeat(x), reversed(range(n)))
    return sumprod(coefficients, powers)

def iter_index(iterable, value, start=0):
    "Return indices where a value occurs in a sequence or iterable"
    # iter_index('aabcdeaf', 'a')  -> [0, 1, 6]
    try: 
        seq_index = iterable.index  # builtin_function_or_method
    except AttributeError:
        # slow path for general iterables 
        it = itertools.islice(iterable, start, None)
        i = start - 1 
        try: 
            while True:
                #s = 'aabcdeaf'
                # it = iter(s)
                # operator.indexOf(it, 'b')  -> 2
                yield (i := 1 + operator.indexOf(it, value) + 1)
        except ValueError:
            pass 
    else:
        # fast path for sequences 
        i = start - 1 
        try: 
            while True:
                # s = 'aabcdeaf'
                # si = s.index
                # si('a',1), si('a', 2)  -> (1,6)
                yield (i := seq_index(value, i+1))
        except ValueError:
            pass 

def sieve(n):
    "Prime less than n"
    # sieve(30) -> 2 3 5 7 11 13 17 19 23 29 
    # bytearray((0, 1)) * (10 // 2)  -> bytearray(b'\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01')
    data = bytearray((0, 1)) * (n // 2)
    data[:3] = 0, 0, 0
    # math.isqrt(n, /), Return the integer part of the square root of the input.
    limit = math.isqrt(n) + 1 
    # compress(data, selectors)
    # Return data elements corresponding to true selector elements
    for p in itertools.compress(range(limit), data):
        data[p*p : n : p+p] = bytes(len(range(p*p, n, p+p)))
    data[2] = 1 
    # use above defined iter_index 
    return iter_index(data, 1) if n > 2 else iter([])

def factor(n):
    "Prime factors of n."
    # factor(99) -> 3 3 11 
    for prime in sieve(math.isqrt(n) + 1):   # get prime numbers 
        while True:
            # divmod(10, 3) -> (3, 1)
            quotient, remainder = divmod(n, prime)
            if remainder:   #  not a divisor 
                break 
            yield prime 
            n = quotient    # continue  to  divide 
            if n == 1:      # exit 
                return 
    if n > 1:    # when n is bigger than 1
        yield n  

def flatten(list_of_lists):
    "Flatten one level of nesting"
    # flatten [1, 2, 3, 4, 5, 6]
    return itertools.chain.from_iterable(list_of_lists)

def repeatfunc(func, times=None, *args):
    """ 
    Repeat calls to func with specified arguments.
    Example: repeatfunc(random.random) -> [0.056636868120843675, 0.9628687773066632]
    """
    if times is None:
        return itertools.starmap(func, itertools.repeat(args))
    return itertools.starmap(func, itertools.repeat(args, times))

def triplewise(iterable):
    "Return overlapping triplets from an iterable"
    # triplewise('abcdefg) -> abc bcd cdde edef efg 
    # list(itertools.pairwise(itertools.pairwise("iterable"))) -> 
    #  [(('i', 't'), ('t', 'e')),
    #  (('t', 'e'), ('e', 'r')),
    #  (('e', 'r'), ('r', 'a')), ...
    #  skip 2nd repeat value, use pairwise twice 
    # itertools.pairwise, was added in python 3.10
    for (a, _), (b, c) in itertools.pairwise(itertools.pairwise(iterable)):
        yield a, b, c 

def sliding_window(iterable, n):
    # sliding_window('abcdefg', 4) -> [('a', 'b', 'c', 'd'), ('b', 'c', 'd', 'e'), ('c', 'd', 'e', 'f'), ('d', 'e', 'f', 'g')]
    # it = iter('abcdefg')
    # n=4
    # window = collections.deque(itertools.islice(it, n), maxlen=n)
    # window   -> deque(['a', 'b', 'c', 'd'], maxlen=4)
    # window.append('f')
    # window   -> deque(['b', 'c', 'd', 'f'], maxlen=4)   # front 'a' removed automatically
    it = iter(iterable)
    window = collections.deque(itertools.islice(it, n), maxlen=n)
    if len(window) == n:    # length is same 
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)

def roundrobin(*iterables):
    "roundrobin('abc', 'd', 'ef') -> a d e b f c"   # ['a', 'd', 'e', 'b', 'f', 'c']
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try: 
            for next in nexts:
                yield next()     # next() function
        except StopIteration:
            # remove the iterator we just exhausted from the cycle  
            num_active -= 1 
            nexts = itertools.cycle(itertools.islice(nexts, num_active))

def partition(pred, iterable):
    "Use a predicate to partition entries into false entries and true entries"
    # partition(is_odd, range(10)) -> 0 2 4 8 and 1 3 5 7 9   # [[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]]
    t1, t2 = itertools.tee(iterable)   # create duplicate iterators 
    return itertools.filterfalse(pred, t1), filter(pred, t2)

def before_and_after(predicate, it):
    """ 
    Variant of takewhile() that allows complete access to the remainer of the iterator.

    it = iter('ABCdEfGhI')
    all_upper, remainder = before_and_after(str.isupper, it)
    ''.join(all_upper)    # -> 'ABC'
    ''.join(remainder)    # -> 'dEfGhI' , takewhile() would lose the 'd'

    Note that the first iterator must be fullly consumed before the second iterator 
    can generate valid results.

    it = iter('ABCdEfGhI')
    all_upper, remainder = before_and_after(str.isupper, it)   # -> ['A', 'B', 'C', 'E', 'G', 'I'] ['d', 'f', 'h']
    """
    it = iter(it)
    transition = [] 
    def true_iterator():
        for elem in it:
            if predicate(elem):
                yield elem 
            else:
                transition.append(elem)   # not losing 'd', carry over to remainder 
    def remainder_iterator(): 
        yield from transition 
        yield from it 
    return true_iterator(), remainder_iterator()

def subslices(seq):
    "Return all contiguous non-empty subslices of a sequence"
    # subslices('abcd') -> a ab abc abcd b bc bcd c cd d   
    # -> ['a', 'ab', 'abc', 'abcd', 'b', 'bc', 'bcd', 'c', 'cd', 'd']
    slices = itertools.starmap(slice, itertools.combinations(range(len(seq)+1), 2))
    return map(operator.getitem, itertools.repeat(seq), slices)

def powerset(iterable):
    "powerset([1,2,3]) -> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)" 
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)) 

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('aaaabbbccdaabbc')  -> a b c d 
    # unique_everseen('ABBcCAD', str.lower) -> A B c D  # ['A', 'B', 'c', 'D']
    seen = set()
    if key is None:
        for element in itertools.filterfalse(seen.__contain__, iterable):
            seen.add(element)
            yield element 
        # For order preserving deduplication, 
        # a faster but non-lazy solution is:
        # yield from dict.fromkeys(iterable)
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen.add(k)
                yield element 
        # For use cases that allow the last matching element to be returned,
        # a faster but non-lazy solution is:
        #     t1, t2 = tee(iterable)
        #     yield from dictt(zip(map(key, t1), t2)).vallues()

def unique_justseen(iterable, key=None):
    "List unique elements, preserving order, remember only the element just seen."
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBcCAD', str.lower) --> A B c A D
    return map(next, map(operator.itemgetter(1), itertools.groupby(iterable, key)))

def iter_except(func, exception, first=None):
    """ 
    Call a function repeatedlly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like builtins.iter(func, sentinnel) but uses an exception instead 
    of a sentinl to end the loop.

   Examples:
        iter_except(functools.partial(heappop, h), IndexError)   # priority queue iterator
        iter_except(d.popitem, KeyError)                         # non-blocking dict iterator
        iter_except(d.popleft, IndexError)                       # non-blocking deque iterator
        iter_except(q.get_nowait, Queue.Empty)                   # loop over a producer Queue
        iter_except(s.pop, KeyError)                             # non-blocking set iterator    
    """
    try: 
        if first is not None:
            yield first()
        while True: 
            yield func()
    except exception:
        pass 

def first_true(iterable, default=False, pred=None):
    """ 
    Return the first true value in the iterable.
    If not true value is found, returns  *default*.
    If *pred* is not None, returns the first item 
    for which pred(item) is true.
    """
    # first_true([a,b,c], x) -> a or b or c or x 
    # first_true([a,b], x, f) -> a if f(a) else b if f(b) else x 
    return next(filter(pred, iterable), default)

def nth_combination(iterable, r, index):
    "Equivalent to list(combinations(iterable, r))[index]"
    # nth_combination('abcdefg', 4, 2) -> ('a', 'b', 'c', 'f')
    pool = tuple(iterable)
    n = len(pool)
    c = math.comb(n, r)
    if index < 0: 
        index += c 
    if index < 0 or index >= c:
        raise IndexError 
    result = [] 
    while r: 
        c, n, r = c*r//n, n-1, r-1 
        while index >= c:
            index -= c 
            c, n = c*(n-r)//n, n-1 
        result.append(pool[-1-n])
    return tuple(result)


# test cases 
# nums = random.choices(range(0,10), k=6)
nums = 7, 1, 6, 8, 5, 2
print(f"{nums=}", take(3, nums))   # nums=[7, 1, 6, 8, 5, 2] [7, 1, 6]
print(f"{nums=}", list(prepend(3, nums)))   # nums=[7, 1, 6, 8, 5, 2] [7, 1, 6]
square = lambda x: x**2 
it_square = tabulate(square, 3)
print(take(4, tabulate(square, 3)))  # [9, 16, 25, 36]
print(f"{nums=}", list(tail(3, nums)))    # nums=[7, 7, 1, 1, 5, 0] [1, 5, 0]
# consume 
i = (x for x in range(10))   # create a generator 
print(next(i))   # -> 0
consume(i, 5)
print(next(i))   # -> 6,  1-5 skipped 

print(f"{nums=}", nth(nums, 3))    # nums=(7, 1, 6, 8, 5, 2) 8
print(f"all_equal", all_equal('aaaab'))    # -> all_equal False

print(f"quantity", quantity('aabb'))    # quantity 4
print(list(ncycles('abc', 3)))    # ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']

print(list(grouper('abcdefg', 3)))    # [('a', 'b', 'c'), ('d', 'e', 'f'), ('g', None, None)]

print(sumprod([1,2], [3,4]))   # -> [(1, 11), (2, 22), (3, 33)]

print(list(transpose([(1,2,3),(11,22,33)])))   # -> [(1, 11), (2, 22), (3, 33)]

print(list(matmul([(1,2), (3, 4)], [[5, 6], [7, 8]])))   # -> [(19, 22), (43, 50)]

print(list(convolve([1,1,1,1], [0.25, 0.25, 0.25, 0.25])))   # -> [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]

print(list(iter_index('aabcdeaf', 'a')))    # [0, 1, 6]

print(list(sieve(30)))  # -> [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

print("factor", list(factor(99)))

print("flatten", list(flatten([[1,2,3],[4,5,6]])))  # flatten [1, 2, 3, 4, 5, 6]

print(list(repeatfunc(random.random, times=2)))   # -> [0.056636868120843675, 0.9628687773066632]

print(list(triplewise("abcdefg")))   # -> [('a', 'b', 'c'), ('b', 'c', 'd'), ('c', 'd', 'e'), ('d', 'e', 'f'), ('e', 'f', 'g')]

print(list(sliding_window("abcdefg", 4)))   # -> [('a', 'b', 'c', 'd'), ('b', 'c', 'd', 'e'), ('c', 'd', 'e', 'f'), ('d', 'e', 'f', 'g')]

print(list(roundrobin('abc', 'd', 'ef')))   # -> ['a', 'd', 'e', 'b', 'f', 'c']

def is_odd(n):
    return n%2 == 0
print([list(i) for i in (partition(is_odd, range(10)))])   # -> [[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]]

it = iter('ABCdEfGhI')
all_upper, remainder = before_and_after(str.isupper, it)
print(list(all_upper), list(remainder))   # -> ['A', 'B', 'C', 'E', 'G', 'I'] ['d', 'f', 'h']

print(list(subslices('abcd')))   # -> ['a', 'ab', 'abc', 'abcd', 'b', 'bc', 'bcd', 'c', 'cd', 'd']

print(list(powerset([1,2,3])))   # [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

print(list(unique_everseen('ABBcCAD', str.lower)))   # ['A', 'B', 'c', 'D']

print(list(unique_justseen('AAAABBBCCDAABBB')))   # ['A', 'B', 'C', 'D', 'A', 'B']

print(list(unique_justseen('ABBcCAD', str.lower)))     # ['A', 'B', 'c', 'A', 'D']

print(list(first_true(['a','b','c'], 'x')))   # -> ['a']

print(nth_combination('abcdefg', 4, 2))   # -> ('a', 'b', 'c', 'f')