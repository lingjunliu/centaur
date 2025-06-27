import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If tensor shape on the first dimension equal to 1 and its dtype not complex64 or complex128, and tensor's ndim smaller than 3, then string v_2 must be equal to 'constant' or equal to 'none' (Rule 877)

rule_877 = lambda s, v, n=False: (
    s.add(Not(If(And(And((Select(v["arg1_shape"], 0) == 1), (And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10))), (v["arg1_ndim"] < 3)), Or((v["arg2_value"] == 10), (v["arg2_value"] == 6)), False)) if n else
          If(And(And((Select(v["arg1_shape"], 0) == 1), (And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10))), (v["arg1_ndim"] < 3)), Or((v["arg2_value"] == 10), (v["arg2_value"] == 6)), False))
)

def rule_877_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 877
        rule_877(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg1_dtype_': arg1_dtype_, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_877(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg1_dtype_': arg1['dtype_'], 'arg2_value': arg2['value']}, neg)
