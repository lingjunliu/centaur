import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the string v_1 is not equal to "min" or "max", then the tensor v_2 should be 3-dimensional (Rule 55)

rule_55 = lambda s, v: (
    s.add(If(And(v["arg1_value"] != "min", v["arg1_value"] != "max"), v["arg2_ndim"] == 3, False))
)

def rule_55_func(arg1, arg2, solver=None):
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

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 55
        rule_55(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']})
