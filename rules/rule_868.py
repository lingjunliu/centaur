import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If string equals to 'bn,anm,bm->ba' then the shapes of v_1,v_2,v_3 has to satisfy shape(v_1,0 (Rule 868)

rule_868 = lambda s, v, n=False: (
    s.add(Not(If((v["arg4_value"] == 5), And((Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), (Select(v["arg2_shape"], 2) == Select(v["arg3_shape"], 0))), False)) if n else
          If((v["arg4_value"] == 5), And((Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), (Select(v["arg2_shape"], 2) == Select(v["arg3_shape"], 0))), False))
)

def rule_868_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not (isinstance(arg3, np.ndarray)):
            return False
        if not (isinstance(arg4, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = String('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == list_of_string_values.index(arg4))

        # Constraints for rule 868
        rule_868(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_868(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value']}, neg)
