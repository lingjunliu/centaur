import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If s[i] is not -1 and input is complex, out needs to be complex (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(If(And(Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11), And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) != -1) for i in range(6)])), Or(v["arg3_dtype"] == 10, v["arg3_dtype"] == 11), True)) if n else
          If(And(Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11), And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) != -1) for i in range(6)])), Or(v["arg3_dtype"] == 10, v["arg3_dtype"] == 11), True))
)

def rule_55_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 55
        rule_55(solver, {'arg1_dtype': arg1_dtype, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_dtype': arg1['dtype'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_dtype': arg3['dtype']}, neg)
