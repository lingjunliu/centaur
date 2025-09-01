import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# dtype must be one of the quantized dtypes: torch.quint8, torch.qint8, torch.qint32, represented by type indices. (Rule 4)

rule_4 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg1_value"] == 5, v["arg1_value"] == 1), v["arg1_value"] == 3)) if n else
          Or(Or(v["arg1_value"] == 5, v["arg1_value"] == 1), v["arg1_value"] == 3))
)

def rule_4_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))

        # Constraints for rule 4
        rule_4(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_4(solver, {'arg1_value': arg1['value']}, neg)
