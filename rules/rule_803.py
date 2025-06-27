import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if number of dimension is equal to 3, then at least one of the tensor's shape should not be equal to min value (Rule 803)

rule_803 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 3, Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) != Select(v["arg1_range"], 0)) for i in range(6)]), False)) if n else
          If(v["arg1_ndim"] == 3, Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) != Select(v["arg1_range"], 0)) for i in range(6)]), False))
)

def rule_803_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 803
        rule_803(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_803(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)
