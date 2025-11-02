import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 is a list of bool, then at least one item must be true and another must be false (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(And((Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) == True) for i in range(6)])), (Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) == False) for i in range(6)])))) if n else
          And((Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) == True) for i in range(6)])), (Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) == False) for i in range(6)]))))
)

def rule_36_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, bool) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), BoolSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 36
        rule_36(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)
