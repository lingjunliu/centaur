import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If number of dimension of tensor v_1 is greater than 1 and minimum number in this tensor is less than 0, then if string v_2 is equal to max then all the shapes must be not equal to 100 (Rule 866)

rule_866 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg1_ndim"] > 1), (Select(v["arg1_range"], 0) < 0)), (v["arg2_value"] == 9)), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) != 100) for i in range(6)]), False)) if n else
          If(And(And((v["arg1_ndim"] > 1), (Select(v["arg1_range"], 0) < 0)), (v["arg2_value"] == 9)), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) != 100) for i in range(6)]), False))
)

def rule_866_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 866
        rule_866(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_866(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
