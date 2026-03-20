import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Dims are non-negative integers and value is a scalar with compatible dtype (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] > 0, And([Implies(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) >= 0, v["arg2_ndim"] == 0), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 0, v["arg2_dtype"] == 1), v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5), v["arg2_dtype"] == 6), v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), v["arg2_dtype"] == 11), v["arg2_dtype"] == 12)))) for i in range(6)]))) if n else
          And(v["arg1_length"] > 0, And([Implies(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) >= 0, v["arg2_ndim"] == 0), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_dtype"] == 0, v["arg2_dtype"] == 1), v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5), v["arg2_dtype"] == 6), v["arg2_dtype"] == 7), v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), v["arg2_dtype"] == 10), v["arg2_dtype"] == 11), v["arg2_dtype"] == 12)))) for i in range(6)])))
)

def rule_27_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 27
        rule_27(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
