import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If bool v_1 is True, float v_2 has shape less than 100 then tensor v_3 has one dimension where its shape is equal to int 5 (Rule 555)

rule_555 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"], v["arg2_value"] < 100), v["arg3_ndim"] > 0), Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) == 5) for i in range(6)]), False)) if n else
          If(And(And(v["arg1_value"], v["arg2_value"] < 100), v["arg3_ndim"] > 0), Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) == 5) for i in range(6)]), False))
)

def rule_555_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (float, np.floating))):
            return False
        if not (isinstance(arg3, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 555
        rule_555(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_555(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)
