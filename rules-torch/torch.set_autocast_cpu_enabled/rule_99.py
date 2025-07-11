import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Check if list contains float elements and the sum of its elements is positive and smaller than a constant AND length is smaller than some number and min value greater than 0 and its datatype should be 8. And each element has to be smaller than max (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_length"] < v["arg3_value"], (Or([And(i < (v["arg1_length"] - 1 + 1), (And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) + Select(v["arg1_values"], j) > 0) for j in range(6)]))) for i in range(6)]))), Select(v["arg1_values"], 0) < v["arg2_value"]), Select(v["arg1_values"], 0) > 0), v["arg4_dtype"] == 8), (And([Implies(z < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], z) < Select(v["arg4_range"], 1)) for z in range(6)])))) if n else
          And(And(And(And(And(v["arg1_length"] < v["arg3_value"], (Or([And(i < (v["arg1_length"] - 1 + 1), (And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) + Select(v["arg1_values"], j) > 0) for j in range(6)]))) for i in range(6)]))), Select(v["arg1_values"], 0) < v["arg2_value"]), Select(v["arg1_values"], 0) > 0), v["arg4_dtype"] == 8), (And([Implies(z < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], z) < Select(v["arg4_range"], 1)) for z in range(6)]))))
)

def rule_99_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), RealSort())
        arg2_value = Real('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_dtype = Int('arg4_dtype')
        arg4_range = Array('arg4_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))

        # Constraints for rule 99
        rule_99(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype, 'arg4_range': arg4_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype'], 'arg4_range': arg4['range']}, neg)
