import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the tensor is not complex and at least one element inside a tuple is bigger than 5 then the maximum of the tensor should be equal to a value greater than 5 (Rule 106)

rule_106 = lambda s, v, n=False: (
    s.add(Not(If(And((And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10)), (Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) > 5) for i in range(6)]))), Select(v["arg1_range"], 1) > 5, True)) if n else
          If(And((And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10)), (Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) > 5) for i in range(6)]))), Select(v["arg1_range"], 1) > 5, True))
)

def rule_106_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 106
        rule_106(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_106(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
