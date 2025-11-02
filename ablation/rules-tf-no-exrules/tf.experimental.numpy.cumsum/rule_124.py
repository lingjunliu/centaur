import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Input a should be of type tensor not dtype (Rule 124)

rule_124 = lambda s, v, n=False: (
    s.add(Not(False) if n else
          False)
)

def rule_124_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 124
        rule_124(solver, {})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_124(solver, {}, neg)
