import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If v_1 threads are used and v_2 is a string then v_2 should not be in ["ii", "ii->i", "i,j->ij", "bij,bjk->bik", "...ij->...ji", "bn,anm,bm->ba"] if v_1 is greater than 1 (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 1, And(And(And(And(And(v["arg2_value"] != 0, v["arg2_value"] != 1), v["arg2_value"] != 2), v["arg2_value"] != 3), v["arg2_value"] != 4), v["arg2_value"] != 5), False)) if n else
          If(v["arg1_value"] > 1, And(And(And(And(And(v["arg2_value"] != 0, v["arg2_value"] != 1), v["arg2_value"] != 2), v["arg2_value"] != 3), v["arg2_value"] != 4), v["arg2_value"] != 5), False))
)

def rule_22_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 22
        rule_22(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
