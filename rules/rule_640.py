import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If a boolean variable is true, then the input tensor must be 3-dimensional (Rule 640)

rule_640 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, v["arg1_ndim"] == 3, False)) if n else
          If(v["arg2_value"] == True, v["arg1_ndim"] == 3, False))
)

def rule_640_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)

        # Constraints for rule 640
        rule_640(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_640(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
