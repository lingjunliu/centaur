import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If int v_1 is less than 0 then each value of the shape of tensor v_2 in each dimension should be less than float number v_3 (Rule 549)

rule_549 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] < 0, v["arg2_ndim"] > 0), And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) < v["arg3_value"]) for i in range(6)]), False)) if n else
          If(And(v["arg1_value"] < 0, v["arg2_ndim"] > 0), And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) < v["arg3_value"]) for i in range(6)]), False))
)

def rule_549_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not (isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == arg3)

        # Constraints for rule 549
        rule_549(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_549(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
