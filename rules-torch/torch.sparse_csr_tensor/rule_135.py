import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If size is a list, elements must be non-negative and less than 2^16 to avoid overflow during allocation AND at least one element in the size should be greater than one and len of the size > 3 and the length should be less than 6 and first two elements should be divisble by 3. (Rule 135)

rule_135 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) >= 0, Select(v["arg1_values"], i) < 65536)) for i in range(6)])), (Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) > 1) for i in range(6)]))), (v["arg1_length"] > 3)), (v["arg1_length"] < 6)), (Select(v["arg1_values"], 0) % 3 == 0)), (Select(v["arg1_values"], 1) % 3 == 0))) if n else
          And(And(And(And(And((And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) >= 0, Select(v["arg1_values"], i) < 65536)) for i in range(6)])), (Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) > 1) for i in range(6)]))), (v["arg1_length"] > 3)), (v["arg1_length"] < 6)), (Select(v["arg1_values"], 0) % 3 == 0)), (Select(v["arg1_values"], 1) % 3 == 0)))
)

def rule_135_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 135
        rule_135(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_135(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)
