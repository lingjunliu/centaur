import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Exactly one of output_size or output_ratio should be specified. Representing it by checking number of dimensions are 3 if it is a tuple (Rule 97)

rule_97 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] == 3, v["arg2_length"] == 0, If(v["arg2_length"] == 3, v["arg1_length"] == 0, False))) if n else
          If(v["arg1_length"] == 3, v["arg2_length"] == 0, If(v["arg2_length"] == 3, v["arg1_length"] == 0, False)))
)

def rule_97_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 97
        rule_97(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_97(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length']}, neg)
