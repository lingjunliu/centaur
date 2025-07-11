import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# dim0 and dim1 cannot be equal, but consider indexing from the end as equivalent to positive indexing when equal (Rule 6)

rule_6 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg1_value"] != v["arg2_value"], (v["arg1_value"] + v["arg3_ndim"] == v["arg2_value"])), (v["arg2_value"] + v["arg3_ndim"] == v["arg1_value"]))) if n else
          Or(Or(v["arg1_value"] != v["arg2_value"], (v["arg1_value"] + v["arg3_ndim"] == v["arg2_value"])), (v["arg2_value"] + v["arg3_ndim"] == v["arg1_value"])))
)

def rule_6_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 6
        rule_6(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_6(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)
