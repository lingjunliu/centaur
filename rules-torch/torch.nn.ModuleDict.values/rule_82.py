import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# if the shape of a tensor in ModuleDict.values is greater than a threshold, the minimum must be greater than another threshold, and it is float type, and dimensions are all positive. Additionally min cannot be max to suppress division by zero and all dimensions must exceed some threshold. If its not true, just return true if the dimension > 3 (Rule 82)

rule_82 = lambda s, v, n=False: (
    s.add(Not(If(And(And(Select(v["arg1_shape"], 0) > v["arg2_value"], v["arg1_dtype"] == 7), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))), (And(And(Select(v["arg1_range"], 0) > v["arg3_value"], Select(v["arg1_range"], 0) != Select(v["arg1_range"], 1)), (And([Implies(k < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], k) > 1) for k in range(6)])))), If(v["arg1_ndim"] > 3, True, False))) if n else
          If(And(And(Select(v["arg1_shape"], 0) > v["arg2_value"], v["arg1_dtype"] == 7), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))), (And(And(Select(v["arg1_range"], 0) > v["arg3_value"], Select(v["arg1_range"], 0) != Select(v["arg1_range"], 1)), (And([Implies(k < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], k) > 1) for k in range(6)])))), If(v["arg1_ndim"] > 3, True, False)))
)

def rule_82_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 82
        rule_82(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_82(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
