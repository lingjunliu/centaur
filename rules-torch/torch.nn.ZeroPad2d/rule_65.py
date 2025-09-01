import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the padding is a tuple, then each element has to be smaller than the maximum size, if the tensor ndim >2 (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_length"] == 4, v["arg2_ndim"] > 2), And(And(And(Select(v["arg1_values"], 0) < 1000, Select(v["arg1_values"], 1) < 1000), Select(v["arg1_values"], 2) < 1000), Select(v["arg1_values"], 3) < 1000), True)) if n else
          If(And(v["arg1_length"] == 4, v["arg2_ndim"] > 2), And(And(And(Select(v["arg1_values"], 0) < 1000, Select(v["arg1_values"], 1) < 1000), Select(v["arg1_values"], 2) < 1000), Select(v["arg1_values"], 3) < 1000), True))
)

def rule_65_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 65
        rule_65(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_ndim': arg2['ndim']}, neg)
