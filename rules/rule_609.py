import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if float v_1 is between 0 and 10, then at least one shape dimension of tensor v_2 must be greater than 5 (Rule 609)

rule_609 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] > 0, v["arg1_value"] < 10), v["arg2_ndim"] > 0), Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 5) for i in range(6)]), False)) if n else
          If(And(And(v["arg1_value"] > 0, v["arg1_value"] < 10), v["arg2_ndim"] > 0), Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 5) for i in range(6)]), False))
)

def rule_609_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating))):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 609
        rule_609(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_609(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
