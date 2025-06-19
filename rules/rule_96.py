import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if the dimension of tensor v_1 is not equal to 3, then the string v_2 must be 'other', else the string v_2 must be equal to '3d' (Rule 96)

rule_96 = lambda s, v: (
    s.add(If(v["arg1_ndim"] != 3, v["arg2_value"] == "other", v["arg2_value"] == "3d"))
)

def rule_96_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)

        # Constraints for rule 96
        rule_96(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_96(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']})
