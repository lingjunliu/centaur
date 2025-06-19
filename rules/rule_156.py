import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If int or float v_1 is positive and bool v_2 is false, then all elements of the string type tensor v_3 must not be empty (Rule 156)

rule_156 = lambda s, v: (
    s.add(If(And(And(v["arg1_value"] > 0, v["arg2_value"] == False), v["arg3_dtype"] == 11), And([Implies(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) != "") for i in range(6)]), False))
)

def rule_156_func(arg1, arg2, arg3, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not (isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 156
        rule_156(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_156(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim']})
