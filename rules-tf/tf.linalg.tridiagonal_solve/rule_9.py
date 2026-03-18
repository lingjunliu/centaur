import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If diagonals_format is "sequence", then diagonals must be a tuple of 3 tensors (Rule 9)
# {v_1 : str, v_2 : tuple(tensor)} |= if v_1 = "sequence" then v_2.len = 3

rule_9 = lambda s, v, n=False: (
    s.add(Not(If(
        v["arg1_val"] == StringVal("sequence"),
        v["arg2_len"] == IntVal(3),
        True
    )) if n else
    If(
        v["arg1_val"] == StringVal("sequence"),
        v["arg2_len"] == IntVal(3),
        True
    ))
)

def rule_9_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, (tuple, list)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_val = String('arg1_val')
        arg2_len = Int('arg2_len')

        # Value assignments
        solver.add(arg1_val == StringVal(arg1))
        solver.add(arg2_len == len(arg2))

        # Constraints for rule 9
        rule_9(solver, {'arg1_val': arg1_val, 'arg2_len': arg2_len})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_val': arg1['val'], 'arg2_len': arg2['len']}, neg)