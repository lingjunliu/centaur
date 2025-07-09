import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the object is a tensor, the product of the shape dimensions must not overflow when multiplied by the size of the tensor datatype. (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 0, Select(v["arg1_shape"], 0) < 2147483647, If(v["arg1_dtype"] == 1, Select(v["arg1_shape"], 0) < 2147483647, If(v["arg1_dtype"] == 2, Select(v["arg1_shape"], 0) < 1073741823, If(v["arg1_dtype"] == 3, Select(v["arg1_shape"], 0) < 536870911, If(v["arg1_dtype"] == 4, Select(v["arg1_shape"], 0) < 268435455, If(v["arg1_dtype"] == 5, Select(v["arg1_shape"], 0) < 2147483647, If(v["arg1_dtype"] == 6, Select(v["arg1_shape"], 0) < 536870911, If(v["arg1_dtype"] == 7, Select(v["arg1_shape"], 0) < 268435455, If(v["arg1_dtype"] == 8, Select(v["arg1_shape"], 0) < 134217727, If(v["arg1_dtype"] == 9, Select(v["arg1_shape"], 0) < 268435455, If(v["arg1_dtype"] == 10, Select(v["arg1_shape"], 0) < 134217727, False)))))))))))) if n else
          If(v["arg1_dtype"] == 0, Select(v["arg1_shape"], 0) < 2147483647, If(v["arg1_dtype"] == 1, Select(v["arg1_shape"], 0) < 2147483647, If(v["arg1_dtype"] == 2, Select(v["arg1_shape"], 0) < 1073741823, If(v["arg1_dtype"] == 3, Select(v["arg1_shape"], 0) < 536870911, If(v["arg1_dtype"] == 4, Select(v["arg1_shape"], 0) < 268435455, If(v["arg1_dtype"] == 5, Select(v["arg1_shape"], 0) < 2147483647, If(v["arg1_dtype"] == 6, Select(v["arg1_shape"], 0) < 536870911, If(v["arg1_dtype"] == 7, Select(v["arg1_shape"], 0) < 268435455, If(v["arg1_dtype"] == 8, Select(v["arg1_shape"], 0) < 134217727, If(v["arg1_dtype"] == 9, Select(v["arg1_shape"], 0) < 268435455, If(v["arg1_dtype"] == 10, Select(v["arg1_shape"], 0) < 134217727, False))))))))))))
)

def rule_44_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 44
        rule_44(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)
