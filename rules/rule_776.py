import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if input v_1 has ndim 2, and its dtype is float, then shape of v_1 in dimension 0 should not be equal shape of v_1 in dimension 1 (Rule 776)

rule_776 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_ndim"] == 2), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8))), Select(v["arg1_shape"], 0) != Select(v["arg1_shape"], 1), False)) if n else
          If(And((v["arg1_ndim"] == 2), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8))), Select(v["arg1_shape"], 0) != Select(v["arg1_shape"], 1), False))
)

def rule_776_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 776
        rule_776(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_776(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_']}, neg)
