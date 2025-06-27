import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if boolean v_1 equals false then for a given dimension v_2 for a tensor v_3, the shape must be one of [2,4,8,16,32,64,128,256,512,1024] (Rule 478)

rule_478 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] == False, v["arg2_value"] >= 0), v["arg2_value"] < v["arg3_ndim"]), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg3_shape"], v["arg2_value"]) == 2, Select(v["arg3_shape"], v["arg2_value"]) == 4), Select(v["arg3_shape"], v["arg2_value"]) == 8), Select(v["arg3_shape"], v["arg2_value"]) == 16), Select(v["arg3_shape"], v["arg2_value"]) == 32), Select(v["arg3_shape"], v["arg2_value"]) == 64), Select(v["arg3_shape"], v["arg2_value"]) == 128), Select(v["arg3_shape"], v["arg2_value"]) == 256), Select(v["arg3_shape"], v["arg2_value"]) == 512), Select(v["arg3_shape"], v["arg2_value"]) == 1024)), False)) if n else
          If(And(And(v["arg1_value"] == False, v["arg2_value"] >= 0), v["arg2_value"] < v["arg3_ndim"]), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg3_shape"], v["arg2_value"]) == 2, Select(v["arg3_shape"], v["arg2_value"]) == 4), Select(v["arg3_shape"], v["arg2_value"]) == 8), Select(v["arg3_shape"], v["arg2_value"]) == 16), Select(v["arg3_shape"], v["arg2_value"]) == 32), Select(v["arg3_shape"], v["arg2_value"]) == 64), Select(v["arg3_shape"], v["arg2_value"]) == 128), Select(v["arg3_shape"], v["arg2_value"]) == 256), Select(v["arg3_shape"], v["arg2_value"]) == 512), Select(v["arg3_shape"], v["arg2_value"]) == 1024)), False))
)

def rule_478_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False
        if not (isinstance(arg3, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 478
        rule_478(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_478(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)
