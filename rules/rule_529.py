import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if string v_1 equals "...ij->...ji", then bool v_2 needs to be false or tensor v_3 shape in the first dimension must be an even number (Rule 529)

rule_529 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 4, Or(v["arg2_value"] == False, (And(Select(v["arg3_shape"], 0) > 0, (Select(v["arg3_shape"], 0) / 2) * 2 == Select(v["arg3_shape"], 0)))), False)) if n else
          If(v["arg1_value"] == 4, Or(v["arg2_value"] == False, (And(Select(v["arg3_shape"], 0) > 0, (Select(v["arg3_shape"], 0) / 2) * 2 == Select(v["arg3_shape"], 0)))), False))
)

def rule_529_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == arg2)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 529
        rule_529(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_529(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)
