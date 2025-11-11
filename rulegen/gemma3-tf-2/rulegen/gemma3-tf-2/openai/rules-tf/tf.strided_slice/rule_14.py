import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# ellipsis_mask has at most one bit set (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 1), v["arg1_value"] == 2), v["arg1_value"] == 4), v["arg1_value"] == 8), v["arg1_value"] == 16), v["arg1_value"] == 32), v["arg1_value"] == 64), v["arg1_value"] == 128), v["arg1_value"] == 256), v["arg1_value"] == 512), v["arg1_value"] == 1024), v["arg1_value"] == 2048), v["arg1_value"] == 4096), v["arg1_value"] == 8192), v["arg1_value"] == 16384), v["arg1_value"] == 32768), v["arg1_value"] == 65536), v["arg1_value"] == 131072), v["arg1_value"] == 262144), v["arg1_value"] == 524288), v["arg1_value"] == 1048576), v["arg1_value"] == 2097152), v["arg1_value"] == 4194304), v["arg1_value"] == 8388608), v["arg1_value"] == 16777216), v["arg1_value"] == 33554432), v["arg1_value"] == 67108864), v["arg1_value"] == 134217728), v["arg1_value"] == 268435456), v["arg1_value"] == 536870912), v["arg1_value"] == 1073741824)) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 1), v["arg1_value"] == 2), v["arg1_value"] == 4), v["arg1_value"] == 8), v["arg1_value"] == 16), v["arg1_value"] == 32), v["arg1_value"] == 64), v["arg1_value"] == 128), v["arg1_value"] == 256), v["arg1_value"] == 512), v["arg1_value"] == 1024), v["arg1_value"] == 2048), v["arg1_value"] == 4096), v["arg1_value"] == 8192), v["arg1_value"] == 16384), v["arg1_value"] == 32768), v["arg1_value"] == 65536), v["arg1_value"] == 131072), v["arg1_value"] == 262144), v["arg1_value"] == 524288), v["arg1_value"] == 1048576), v["arg1_value"] == 2097152), v["arg1_value"] == 4194304), v["arg1_value"] == 8388608), v["arg1_value"] == 16777216), v["arg1_value"] == 33554432), v["arg1_value"] == 67108864), v["arg1_value"] == 134217728), v["arg1_value"] == 268435456), v["arg1_value"] == 536870912), v["arg1_value"] == 1073741824))
)

def rule_14_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 14
        rule_14(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_value': arg1['value']}, neg)
