import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel_size, stride and padding in tuple form should be greater than zero and padding non-negative (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(And(And((And([Implies(v_4 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_4) > 0) for v_4 in range(6)])), (And([Implies(v_5 < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], v_5) > 0) for v_5 in range(6)]))), (And([Implies(v_6 < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], v_6) >= 0) for v_6 in range(6)])))) if n else
          And(And((And([Implies(v_4 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_4) > 0) for v_4 in range(6)])), (And([Implies(v_5 < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], v_5) > 0) for v_5 in range(6)]))), (And([Implies(v_6 < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], v_6) >= 0) for v_6 in range(6)]))))
)

def rule_48_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 48
        rule_48(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_values': arg3_values, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length']}, neg)
