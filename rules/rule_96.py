import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# String value has to be present within Tensor value (Rule 96)

rule_96 = lambda s, v: (
    s.add(If(v["arg2_dtype"] == 11, Or([And(i < (v["arg2_ndim"] - 1 + 1), v["arg1_value"] == Select(v["arg2_shape"], i)) for i in range(6)]), True))
)

def rule_96_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 96
        rule_96(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_96(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']})
