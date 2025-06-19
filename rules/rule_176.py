import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If float v_1 is less than the shape of tensor v_2 at dimension 0, then the data type of tensor v_2 must not be boolean (Rule 176)

rule_176 = lambda s, v: (
    s.add(If(v["arg1_value"] < Select(v["arg2_shape"], 0), v["arg2_dtype"] != 0, False))
)

def rule_176_func(arg1, arg2, solver=None):
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

        # Constraints for rule 176
        rule_176(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_176(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype']})
