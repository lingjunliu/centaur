import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# Given a string v_1 and a tensor v_2, then if v_1 string is equal to 'tanh' then the shape of every dimension in v_2 tensor should be greater or equal to integer 1 (Rule 435)

rule_435 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 11, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) >= 1) for i in range(6)]), False)) if n else
          If(v["arg1_value"] == 11, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) >= 1) for i in range(6)]), False))
)

def rule_435_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 435
        rule_435(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_435(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
