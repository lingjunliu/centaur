import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# For all dimensions > 2 all shape values need to be greater than maximum value when dtype is float32 or float 64 and minimum shape has to be less than that (Rule 200)

rule_200 = lambda s, v: (
    s.add(If(And((v["arg1_ndim"] > 2), (Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 8)))), And((And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > Select(v["arg1_range"], 1)) for i in range(6)])), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < Select(v["arg1_range"], 1)) for i in range(6)]))), True))
)

def rule_200_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 200
        rule_200(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_200(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape']})
