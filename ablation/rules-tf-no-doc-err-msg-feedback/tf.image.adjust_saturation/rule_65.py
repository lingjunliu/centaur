import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# A saturation factor that's too high might lead to an image whose pixel values get clipped because they exceed the maximum allowed value for that data type, while a small saturation factor will lead to washed out colours. (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(And(-0.5 < v["arg1_value"], v["arg1_value"] < 2.5)) if n else
          And(-0.5 < v["arg1_value"], v["arg1_value"] < 2.5))
)

def rule_65_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 65
        rule_65(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_value': arg1['value']}, neg)
