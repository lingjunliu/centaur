import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the dtype of tensor v_1 is in the range of 1 to 5, the number of dimension of the tensor v_2 must be great than 0 and the max shape of tensor v_2 at the first dimension must not be equal to minimum value of tensor v_1 (Rule 192)

rule_192 = lambda s, v: (
    s.add(If(And((And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5)), v["arg2_ndim"] > 0), Select(v["arg2_shape"], 0) != Select(v["arg1_range"], 0), False))
)

def rule_192_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 192
        rule_192(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_192(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']})
