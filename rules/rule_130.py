import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If int v_1 is smaller than 5 and bool v_2 is false, and the number of dimensions of tensor v_3 is more than 1, then the shape on the first dimension and second dimension of tensor v_3 must be equal or not equal (Rule 130)

rule_130 = lambda s, v: (
    s.add(If(And(And(v["arg1_value"] < 5, v["arg2_value"] == False), v["arg3_ndim"] > 1), Or(Select(v["arg3_shape"], 0) == Select(v["arg3_shape"], 1), Select(v["arg3_shape"], 0) != Select(v["arg3_shape"], 1)), False))
)

def rule_130_func(arg1, arg2, arg3, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False
        if not (isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 130
        rule_130(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_130(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']})
