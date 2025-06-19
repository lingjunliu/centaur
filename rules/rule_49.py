import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If boolean v_1 is false and tensor v_2 is complex, then there must be at least one dimension which has a size of at least 2 (Rule 49)

rule_49 = lambda s, v: (
    s.add(If(And(v["arg1_value"] == False, (Or(v["arg2_dtype"] == 9, v["arg2_dtype"] == 10))), Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) >= 2) for i in range(6)]), False))
)

def rule_49_func(arg1, arg2, solver=None):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 49
        rule_49(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']})
