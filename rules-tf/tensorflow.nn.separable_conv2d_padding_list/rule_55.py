import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the tensor is a vector, then one list padding is zero (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_ndim"] == 1, And(And(Select(v["arg1_values"], 0) == 0, Or(Select(v["arg1_values"], 1) == 0, Select(v["arg2_values"], 0) == 0)), Select(v["arg2_values"], 1) == 0), False)) if n else
          If(v["arg3_ndim"] == 1, And(And(Select(v["arg1_values"], 0) == 0, Or(Select(v["arg1_values"], 1) == 0, Select(v["arg2_values"], 0) == 0)), Select(v["arg2_values"], 1) == 0), False))
)

def rule_55_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 55
        rule_55(solver, {'arg1_values': arg1_values, 'arg2_values': arg2_values, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_values': arg1['values'], 'arg2_values': arg2['values'], 'arg3_ndim': arg3['ndim']}, neg)
