import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If a tensor has at least 2 dimensions and data type is float, then the ratio of the largest shape to the smallest shape should be less than 100 (Rule 928)

rule_928 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_ndim"] >= 2), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8))), And((And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_range"], 1) / Select(v["arg1_shape"], i) < 100) for i in range(6)])), (And([Implies(j < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], j) / Select(v["arg1_range"], 0) < 100) for j in range(6)]))), False)) if n else
          If(And((v["arg1_ndim"] >= 2), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8))), And((And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_range"], 1) / Select(v["arg1_shape"], i) < 100) for i in range(6)])), (And([Implies(j < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], j) / Select(v["arg1_range"], 0) < 100) for j in range(6)]))), False))
)

def rule_928_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 928
        rule_928(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_928(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_']}, neg)
