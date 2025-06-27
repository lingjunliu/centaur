import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If tensor v_1's shape at index 0 is greater than float v_2 and v_3 boolean is true, then shape v_1's last dimension has to be one of the supported output shapes (Rule 537)

rule_537 = lambda s, v, n=False: (
    s.add(Not(If(And(And(Select(v["arg1_shape"], 0) > v["arg2_value"], v["arg1_ndim"] > 0), v["arg3_value"]), Or(Or(Or(Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 2), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 4), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 8), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 16), False)) if n else
          If(And(And(Select(v["arg1_shape"], 0) > v["arg2_value"], v["arg1_ndim"] > 0), v["arg3_value"]), Or(Or(Or(Or(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 2), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 4), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 8), Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 16), False))
)

def rule_537_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, (float, np.floating))):
            return False
        if not (isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 537
        rule_537(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_537(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
