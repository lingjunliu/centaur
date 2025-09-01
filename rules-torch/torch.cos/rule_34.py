import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the output type is Short, then the input type must allow safe casting to short after the cosine operation. This includes Int8, Int16, or a floating-point number that, when converted to Short after taking cosine, does not lead to overflow. (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 2, And((Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), If(v["arg2_dtype"] == 2, (And(Select(v["arg1_range"], 0) > -100000, Select(v["arg1_range"], 1) < 100000)), True)), True)) if n else
          If(v["arg2_dtype"] == 2, And((Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), If(v["arg2_dtype"] == 2, (And(Select(v["arg1_range"], 0) > -100000, Select(v["arg1_range"], 1) < 100000)), True)), True))
)

def rule_34_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 34
        rule_34(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype']}, neg)
