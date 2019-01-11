import time
from numba import jit

@jit
def fib_fast(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib_fast(n-1)+fib_fast(n-2)

def fib_slow(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib_slow(n-1)+fib_slow(n-2)

t = time.time()
fib_fast(40)
t_fast = time.time() - t
t = time.time()
fib_slow(40)
t_slow = time.time() - t

print('with numba: ',t_fast)
print('without numba: ',t_slow)
