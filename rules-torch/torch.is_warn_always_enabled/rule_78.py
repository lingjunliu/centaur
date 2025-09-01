import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# list of bools and a tensor: the number of dimensions should less or equal to 3 if any of the bools is true, other wise number of dimensions should be greater than 3 (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(Or((Or([And(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) == True, v["arg2_ndim"] <= 3)) for i in range(6)])), (And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) == False, v["arg2_ndim"] > 3)) for i in range(6)])))) if n else
          Or((Or([And(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) == True, v["arg2_ndim"] <= 3)) for i in range(6)])), (And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) == False, v["arg2_ndim"] > 3)) for i in range(6)]))))
)

def rule_78_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, bool) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), BoolSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 78
        rule_78(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_ndim': arg2['ndim']}, neg)
