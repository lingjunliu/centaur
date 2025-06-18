import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the data type is float and a tensor has dimension equal to 3, then all the shapes dimension needs to be same (Rule 153)

rule_153 = lambda s, v: (
    s.add(If(And((And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8)), (v["arg1_ndim"] == 3)), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], i)) for i in range(6)]), True))
)

def rule_153_func(arg1, solver=None):
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

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 153
        rule_153(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_153(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']})
