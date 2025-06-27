import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# The input v_1 must be a tensor with 2 or more dimensions of float, double, cfloat or cdouble types (Rule 990)

rule_990 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] >= 2, (Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)))) if n else
          And(v["arg1_ndim"] >= 2, (Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))))
)

def rule_990_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 990
        rule_990(solver, {'arg1_dtype_': arg1_dtype_, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_990(solver, {'arg1_dtype_': arg1['dtype_'], 'arg1_ndim': arg1['ndim']}, neg)
