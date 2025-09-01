import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel_size and output_size should satisfy the constraint, tuple case. with shape checks and correct MULOP and check that shape(v_2,1 (Rule 111)

rule_111 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And((Or(v["arg2_ndim"] == 3, v["arg2_ndim"] == 4)), Select(v["arg2_shape"], 0) % (Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1)) == 0), (Select(v["arg2_shape"], 1) * (Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1))) / Select(v["arg2_shape"], 0) > 0), (Select(v["arg2_shape"], 1) * (Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1))) / Select(v["arg2_shape"], 0) == Select(v["arg3_values"], 0) * Select(v["arg3_values"], 1)), Select(v["arg2_shape"], 1) > 0), Select(v["arg1_values"], 0) > 0), Select(v["arg1_values"], 1) > 0), Select(v["arg2_shape"], 2) > 0), (Or(v["arg2_ndim"] < 4, Select(v["arg2_shape"], 3) > 0))), ((Select(v["arg2_shape"], 2) + 2 * Select(v["arg4_values"], 0) - (Select(v["arg1_values"], 0) - 1) - 1) / Select(v["arg1_values"], 0) + 1) > 0), ((Select(v["arg2_shape"], 3) + 2 * Select(v["arg4_values"], 1) - (Select(v["arg1_values"], 1) - 1) - 1) / Select(v["arg1_values"], 1) + 1) > 0)) if n else
          And(And(And(And(And(And(And(And(And(And((Or(v["arg2_ndim"] == 3, v["arg2_ndim"] == 4)), Select(v["arg2_shape"], 0) % (Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1)) == 0), (Select(v["arg2_shape"], 1) * (Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1))) / Select(v["arg2_shape"], 0) > 0), (Select(v["arg2_shape"], 1) * (Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1))) / Select(v["arg2_shape"], 0) == Select(v["arg3_values"], 0) * Select(v["arg3_values"], 1)), Select(v["arg2_shape"], 1) > 0), Select(v["arg1_values"], 0) > 0), Select(v["arg1_values"], 1) > 0), Select(v["arg2_shape"], 2) > 0), (Or(v["arg2_ndim"] < 4, Select(v["arg2_shape"], 3) > 0))), ((Select(v["arg2_shape"], 2) + 2 * Select(v["arg4_values"], 0) - (Select(v["arg1_values"], 0) - 1) - 1) / Select(v["arg1_values"], 0) + 1) > 0), ((Select(v["arg2_shape"], 3) + 2 * Select(v["arg4_values"], 1) - (Select(v["arg1_values"], 1) - 1) - 1) / Select(v["arg1_values"], 1) + 1) > 0))
)

def rule_111_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_values = Array('arg4_values', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])

        # Constraints for rule 111
        rule_111(solver, {'arg1_values': arg1_values, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_values': arg3_values, 'arg4_values': arg4_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_values': arg1['values'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_values': arg3['values'], 'arg4_values': arg4['values']}, neg)
