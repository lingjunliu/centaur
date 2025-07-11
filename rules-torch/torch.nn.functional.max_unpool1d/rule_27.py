import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Check for a specific negative bound in output_size dim 0 (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_values"], 0) >= -1495561033660047044, Select(v["arg1_values"], 0) <= -6947555186400635896)) if n else
          And(Select(v["arg1_values"], 0) >= -1495561033660047044, Select(v["arg1_values"], 0) <= -6947555186400635896))
)

def rule_27_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 27
        rule_27(solver, {'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_values': arg1['values']}, neg)
