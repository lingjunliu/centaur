import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the length of list is greater than 5 and the dtype of a tensor is one of [1,2,3,4,5], and the first element is smaller than the last element, then the first and second elements must have same values. (Rule 118)

rule_118 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_length"] > 5, (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))), Select(v["arg1_values"], 0) < Select(v["arg1_values"], v["arg1_length"] - 1)), Select(v["arg1_values"], 0) == Select(v["arg1_values"], 1), True)) if n else
          If(And(And(v["arg1_length"] > 5, (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))), Select(v["arg1_values"], 0) < Select(v["arg1_values"], v["arg1_length"] - 1)), Select(v["arg1_values"], 0) == Select(v["arg1_values"], 1), True))
)

def rule_118_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 118
        rule_118(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_118(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_dtype': arg2['dtype']}, neg)
