import numpy as np
import torch
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# tf.bitcast shape rule:
# When casting from a smaller dtype to a larger dtype (src_size < dst_size),
# the last dimension of the input must be divisible by (dst_size / src_size).
# Error example: bitcast from qint8 (1 byte) to float32 (4 bytes) with shape [3,2]:
#   last dim 2 is not divisible by 4 → ValueError

dtype_itemsize = {
    0: 1,   # bool
    1: 1,   # int8
    2: 2,   # int16
    3: 4,   # int32
    4: 8,   # int64
    5: 1,   # uint8
    6: 2,   # float16
    7: 4,   # float32
    8: 8,   # float64
    9: 8,   # complex64
    10: 16, # complex128
}

rule_108 = lambda s, v, n=False: (
    s.add(
        Not(
            Or(
                v["src_size"] >= v["dst_size"],
                v["arg1_last_dim"] % (v["dst_size"] / v["src_size"]) == 0
            )
        )
    ) if n else
    s.add(
        Or(
            v["src_size"] >= v["dst_size"],
            v["arg1_last_dim"] % (v["dst_size"] / v["src_size"]) == 0
        )
    )
)

def rule_108_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if arg1.ndim == 0:
            return False

        src_dtype_idx = list_of_available_dtypes.index(arg1.dtype)
        src_size = dtype_itemsize.get(src_dtype_idx, 1)
        dst_size = np.dtype(arg2).itemsize
        last_dim = arg1.shape[-1]

        solver = Solver()
        src_size_var = Int('src_size')
        dst_size_var = Int('dst_size')
        arg1_last_dim = Int('arg1_last_dim')

        solver.add(src_size_var == src_size)
        solver.add(dst_size_var == dst_size)
        solver.add(arg1_last_dim == last_dim)

        rule_108(solver, {
            'src_size': src_size_var,
            'dst_size': dst_size_var,
            'arg1_last_dim': arg1_last_dim
        })
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {
            'src_size': arg1['src_size'],
            'dst_size': arg1['dst_size'],
            'arg1_last_dim': arg1['arg1_last_dim']
        }, neg)