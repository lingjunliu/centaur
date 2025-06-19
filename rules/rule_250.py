import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If v_1 is a tensor with ndim > 0, then v_2 can be string “int”, “float”, “bool”, or “str” only (Rule 250)

rule_250 = lambda s, v: (
    s.add(If(v["arg1_ndim"] > 0, Or(Or(Or(v["arg2_value"] == "int", v["arg2_value"] == "float"), v["arg2_value"] == "bool"), v["arg2_value"] == "str"), False))
)

def rule_250_func(arg1, arg2, solver=None):
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

        # Constraints for rule 250
        rule_250(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_250(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']})
