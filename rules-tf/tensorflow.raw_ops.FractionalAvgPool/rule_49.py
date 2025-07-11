import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Pooling ratios should not be excessively close to 1.0 to avoid minimal pooling (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (2 + 1), Or(Select(v["arg1_values"], i) > 1.05, Select(v["arg1_values"], i) < 0.95)) for i in range(6)])) if n else
          And([Implies(i < (2 + 1), Or(Select(v["arg1_values"], i) > 1.05, Select(v["arg1_values"], i) < 0.95)) for i in range(6)]))
)

def rule_49_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), RealSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 49
        rule_49(solver, {'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_values': arg1['values']}, neg)
