import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# out tensor dtype should be smaller or same as input type and output is floating point if input is floating point, short type for input needs short or higher for output and if input complex, output is complex or bigger and bool can be type promoted to int/float/complex and if input is int8, then output can be int8 to float32 or float64 (Rule 121)

rule_121 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 2, v["arg2_dtype"] >= 2, If(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8), If(And(10 <= v["arg1_dtype"], v["arg1_dtype"] <= 11), v["arg2_dtype"] >= 10, If(v["arg1_dtype"] == 0, v["arg2_dtype"] >= 1, If(v["arg1_dtype"] == 1, (Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 6), v["arg2_dtype"] == 7), v["arg2_dtype"] == 8)), v["arg2_dtype"] <= v["arg1_dtype"])))))) if n else
          If(v["arg1_dtype"] == 2, v["arg2_dtype"] >= 2, If(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8), If(And(10 <= v["arg1_dtype"], v["arg1_dtype"] <= 11), v["arg2_dtype"] >= 10, If(v["arg1_dtype"] == 0, v["arg2_dtype"] >= 1, If(v["arg1_dtype"] == 1, (Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 6), v["arg2_dtype"] == 7), v["arg2_dtype"] == 8)), v["arg2_dtype"] <= v["arg1_dtype"]))))))
)

def rule_121_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 121
        rule_121(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_121(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
