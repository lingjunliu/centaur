import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check sampleRate against most known values (Rule 150)

rule_150 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 8000, v["arg1_value"] == 11025), v["arg1_value"] == 16000), v["arg1_value"] == 22050), v["arg1_value"] == 44100), v["arg1_value"] == 48000), v["arg1_value"] == 96000), v["arg1_value"] == 192000)) if n else
          Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 8000, v["arg1_value"] == 11025), v["arg1_value"] == 16000), v["arg1_value"] == 22050), v["arg1_value"] == 44100), v["arg1_value"] == 48000), v["arg1_value"] == 96000), v["arg1_value"] == 192000))
)

def rule_150_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 150
        rule_150(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_150(solver, {'arg1_value': arg1['value']}, neg)
