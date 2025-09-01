import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Ultimate rule checking all input conditions with existential quantifier (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And((v["arg1_dtype"] != 0), (v["arg2_value"] >= 1)), (v["arg3_length"] == 2)), (v["arg1_dtype"] == 12)), (v["arg1_ndim"] > 0)), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))), And([Implies(j < (v["arg3_length"] - 1 + 1), True) for j in range(6)]))) if n else
          And(And(And(And(And(And((v["arg1_dtype"] != 0), (v["arg2_value"] >= 1)), (v["arg3_length"] == 2)), (v["arg1_dtype"] == 12)), (v["arg1_ndim"] > 0)), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))), And([Implies(j < (v["arg3_length"] - 1 + 1), True) for j in range(6)])))
)

def rule_24_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 24
        rule_24(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)
