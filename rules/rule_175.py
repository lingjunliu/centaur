import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If float v_1 is greater than max shape dimension, then dype of tensor v_2 cannot be int64,complex64,complex128,str or bool (Rule 175)

rule_175 = lambda s, v: (
    s.add(If(v["arg1_value"] > Select(v["arg2_shape"], 0), And(And(And(And(v["arg2_dtype"] != 4, v["arg2_dtype"] != 9), v["arg2_dtype"] != 10), v["arg2_dtype"] != 11), v["arg2_dtype"] != 0), False))
)

def rule_175_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating))):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 175
        rule_175(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_175(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype']})
