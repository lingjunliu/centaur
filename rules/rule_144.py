import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the value is an integer or float and less than zero, then tensor's shape has to be empty on any dimension (Rule 144)

rule_144 = lambda s, v: (
    s.add(If(v["arg1_value"] < 0, (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 0) for i in range(6)])), True))
)

def rule_144_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 144
        rule_144(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_144(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']})
