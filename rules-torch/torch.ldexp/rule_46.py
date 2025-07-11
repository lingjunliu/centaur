import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# if input is float16, output must be float16 or float32, if input is complex64 output must be complex64 or complex128 and the same as the input if it's not a floating point (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 7, (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8)), If(v["arg1_dtype"] == 9, (Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10)), If(v["arg1_dtype"] == 8, (v["arg2_dtype"] == 8), If(v["arg1_dtype"] == 10, (v["arg2_dtype"] == 10), (v["arg1_dtype"] == v["arg2_dtype"])))))) if n else
          If(v["arg1_dtype"] == 7, (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8)), If(v["arg1_dtype"] == 9, (Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10)), If(v["arg1_dtype"] == 8, (v["arg2_dtype"] == 8), If(v["arg1_dtype"] == 10, (v["arg2_dtype"] == 10), (v["arg1_dtype"] == v["arg2_dtype"]))))))
)

def rule_46_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 46
        rule_46(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
