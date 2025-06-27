import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If str v_1 contains substring 'jk' and boolean v_2 is true, then the shapes of last 2 dimensions for the tensor v_3 must be greater than integer 1 (Rule 581)

rule_581 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg1_value"] == 3), (v["arg3_ndim"] >= 2)), v["arg2_value"]), And(Select(v["arg3_shape"], v["arg3_ndim"] - 1) > 1, Select(v["arg3_shape"], v["arg3_ndim"] - 2) > 1), False)) if n else
          If(And(And((v["arg1_value"] == 3), (v["arg3_ndim"] >= 2)), v["arg2_value"]), And(Select(v["arg3_shape"], v["arg3_ndim"] - 1) > 1, Select(v["arg3_shape"], v["arg3_ndim"] - 2) > 1), False))
)

def rule_581_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 581
        rule_581(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_581(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)
