import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# enabled-stable equality under a bound dependent on enabled (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (If(v["arg1_value"], 0, 0) + 1), v["arg1_value"] == v["arg1_value"]) for i in range(6)])) if n else
          And([Implies(i < (If(v["arg1_value"], 0, 0) + 1), v["arg1_value"] == v["arg1_value"]) for i in range(6)]))
)

def rule_9_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 9
        rule_9(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_value': arg1['value']}, neg)
