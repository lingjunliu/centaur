import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel_size, output_size, dilation, padding and stride should satisfy the equation given in the API documentation for tuple case (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_values"], 0) == ((Select(v["arg5_values"], 0) + 2 * Select(v["arg4_values"], 0) - Select(v["arg3_values"], 0) * (Select(v["arg2_values"], 0) - 1) - 1) / Select(v["arg3_values"], 0)) + 1, Select(v["arg1_values"], 1) == ((Select(v["arg5_values"], 1) + 2 * Select(v["arg4_values"], 1) - Select(v["arg3_values"], 1) * (Select(v["arg2_values"], 1) - 1) - 1) / Select(v["arg3_values"], 1)) + 1)) if n else
          And(Select(v["arg1_values"], 0) == ((Select(v["arg5_values"], 0) + 2 * Select(v["arg4_values"], 0) - Select(v["arg3_values"], 0) * (Select(v["arg2_values"], 0) - 1) - 1) / Select(v["arg3_values"], 0)) + 1, Select(v["arg1_values"], 1) == ((Select(v["arg5_values"], 1) + 2 * Select(v["arg4_values"], 1) - Select(v["arg3_values"], 1) * (Select(v["arg2_values"], 1) - 1) - 1) / Select(v["arg3_values"], 1)) + 1))
)

def rule_69_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_values = Array('arg5_values', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])

        # Constraints for rule 69
        rule_69(solver, {'arg1_values': arg1_values, 'arg2_values': arg2_values, 'arg3_values': arg3_values, 'arg4_values': arg4_values, 'arg5_values': arg5_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_values': arg1['values'], 'arg2_values': arg2['values'], 'arg3_values': arg3['values'], 'arg4_values': arg4['values'], 'arg5_values': arg5['values']}, neg)
