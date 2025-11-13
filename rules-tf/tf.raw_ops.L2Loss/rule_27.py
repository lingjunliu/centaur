import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If input tensor `t` contains `bfloat16`, `half`, `float32` or `float64`, then the values must be within a valid range to prevent overflows when calculating the L2 norm. (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, And(Select(v["arg1_range"], 0) > -65504, Select(v["arg1_range"], 1) < 65504), If(v["arg1_dtype"] == 7, And(Select(v["arg1_range"], 0) > -3.4e38, Select(v["arg1_range"], 1) < 3.4e38), If(v["arg1_dtype"] == 8, And(Select(v["arg1_range"], 0) > -1.7e308, Select(v["arg1_range"], 1) < 1.7e308), True)))) if n else
          If(v["arg1_dtype"] == 6, And(Select(v["arg1_range"], 0) > -65504, Select(v["arg1_range"], 1) < 65504), If(v["arg1_dtype"] == 7, And(Select(v["arg1_range"], 0) > -3.4e38, Select(v["arg1_range"], 1) < 3.4e38), If(v["arg1_dtype"] == 8, And(Select(v["arg1_range"], 0) > -1.7e308, Select(v["arg1_range"], 1) < 1.7e308), True))))
)

def rule_27_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 27
        rule_27(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
