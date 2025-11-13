import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# aspect_ratio_range values must be valid and satisfy the relation (Rule 137)

rule_137 = lambda s, v, n=False: (
    s.add(Not(And(And(And((v["arg1_length"] == 2), (Select(v["arg1_values"], 0) >= 0.75)), (Select(v["arg1_values"], 1) <= 1.33)), (Select(v["arg1_values"], 0) < Select(v["arg1_values"], 1)))) if n else
          And(And(And((v["arg1_length"] == 2), (Select(v["arg1_values"], 0) >= 0.75)), (Select(v["arg1_values"], 1) <= 1.33)), (Select(v["arg1_values"], 0) < Select(v["arg1_values"], 1))))
)

def rule_137_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 137
        rule_137(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_137(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)
