import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If the tensor v_1 has three dimensions, then the max element in tensor should be greater than multiplication of its shapes divided by 2 and its type can be either int or float. (Rule 796)

rule_796 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 3, And(Select(v["arg1_range"], 1) > (Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2)) / 2, (And(And(1 <= v["arg1_dtype"], Or(v["arg1_dtype"] <= 5, 6 <= v["arg1_dtype"])), v["arg1_dtype"] <= 8))), False)) if n else
          If(v["arg1_ndim"] == 3, And(Select(v["arg1_range"], 1) > (Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2)) / 2, (And(And(1 <= v["arg1_dtype"], Or(v["arg1_dtype"] <= 5, 6 <= v["arg1_dtype"])), v["arg1_dtype"] <= 8))), False))
)

def rule_796_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 796
        rule_796(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_796(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_']}, neg)
