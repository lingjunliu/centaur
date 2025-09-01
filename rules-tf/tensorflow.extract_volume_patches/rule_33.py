import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If input is uint32, strides and ksizes must also be uint32 compatible. (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 21, And((And([Implies(i < (v["arg2_length"] - 1 + 1), And(Select(v["arg2_values"], i) >= 0, Select(v["arg2_values"], i) <= 4294967295)) for i in range(6)])), (And([Implies(i < (v["arg3_length"] - 1 + 1), And(Select(v["arg3_values"], i) >= 0, Select(v["arg3_values"], i) <= 4294967295)) for i in range(6)]))), True)) if n else
          If(v["arg1_dtype"] == 21, And((And([Implies(i < (v["arg2_length"] - 1 + 1), And(Select(v["arg2_values"], i) >= 0, Select(v["arg2_values"], i) <= 4294967295)) for i in range(6)])), (And([Implies(i < (v["arg3_length"] - 1 + 1), And(Select(v["arg3_values"], i) >= 0, Select(v["arg3_values"], i) <= 4294967295)) for i in range(6)]))), True))
)

def rule_33_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 33
        rule_33(solver, {'arg1_dtype': arg1_dtype, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_dtype': arg1['dtype'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)
