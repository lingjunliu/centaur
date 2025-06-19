import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if bool variable v_1 is true, and dtype of tensor v_2 is a floating point type, then v_2 must have shape at index 0 greater than int variable v_3 (Rule 191)

rule_191 = lambda s, v: (
    s.add(If(And(v["arg1_value"], (And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 8))), Select(v["arg2_shape"], 0) > v["arg3_value"], False))
)

def rule_191_func(arg1, arg2, arg3, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 191
        rule_191(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_191(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']})
