#%% importy

from diskcache import Cache, FanoutCache

#%%

cache = FanoutCache('tmp')

#%%

@cache.memoize(typed=True, expire=1, tag='fib')
def fibonacci(n):

    if n == 0:
        return 1
    if n == 1:
        return 1

    return fibonacci(n-1) + fibonacci(n-2)
# %%
