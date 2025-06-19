import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If number of dimensions for the tensor v_1 is equal to 0, then, data type of the tensor v_1 has to be str (Rule 172)

rule_172 = lambda s, v: (
    s.add(If(v["arg1_ndim"] == 0, v["arg1_dtype"] == 11, False))
)

def rule_172_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 172
        rule_172(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_172(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']})
