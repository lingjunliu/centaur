import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If string v_1 is either 'sum' or 'mean' or 'max' and tensor v_2 is not empty and the integer v_3 is smaller than the dimension of the tensor, then shape of v_2 at v_3 must not be equal to 1 (Rule 481)

rule_481 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And((Or(Or(v["arg1_value"] == 8, v["arg1_value"] == 7), v["arg1_value"] == 9)), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), v["arg3_value"] < v["arg2_ndim"]), v["arg3_value"] >= 0), Select(v["arg2_shape"], v["arg3_value"]) != 1, False)) if n else
          If(And(And(And((Or(Or(v["arg1_value"] == 8, v["arg1_value"] == 7), v["arg1_value"] == 9)), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), v["arg3_value"] < v["arg2_ndim"]), v["arg3_value"] >= 0), Select(v["arg2_shape"], v["arg3_value"]) != 1, False))
)

def rule_481_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 481
        rule_481(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_481(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
