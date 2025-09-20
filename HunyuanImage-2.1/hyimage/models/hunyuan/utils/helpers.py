import collections.abc
from itertools import repeat

def _ntuple(n):
    """
    Returns a function that converts input to a tuple of length n.
    If input is an iterable (except str), it is converted to a tuple.
    If the tuple has length 1, it is repeated n times.
    Otherwise, the input is repeated n times to form the tuple.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
