import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If a bool is false and then dtype should be integer 1 to 5, shape's first value must be even (Rule 167)

rule_167 = lambda s, v: (
    s.add(If(And(v["arg1_value"] == False, (And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 5))), (Select(v["arg2_shape"], 0) * 0.5) == (Select(v["arg2_shape"], 0) / 2), True))
)

def rule_167_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 167
        rule_167(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_167(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']})
