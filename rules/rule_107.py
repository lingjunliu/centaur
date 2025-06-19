import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if the dtype of tensor v_1 is either int8 or uint8, then string v_2 must be assigned to 'byte' (Rule 107)

rule_107 = lambda s, v: (
    s.add(If(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), v["arg2_value"] == "byte", False))
)

def rule_107_func(arg1, arg2, solver=None):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 107
        rule_107(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_107(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']})
