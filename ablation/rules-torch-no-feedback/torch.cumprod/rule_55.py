import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the input dtype is integral, and a dtype is provided, it can only be integral of same or larger rank. If an out tensor is also specified, it can also not violate this. (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), And((And(v["arg1_dtype"] <= v["arg2_value"], v["arg2_value"] <= 5)), (And(v["arg1_dtype"] <= v["arg3_dtype"], v["arg3_dtype"] <= 5))), True)) if n else
          If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), And((And(v["arg1_dtype"] <= v["arg2_value"], v["arg2_value"] <= 5)), (And(v["arg1_dtype"] <= v["arg3_dtype"], v["arg3_dtype"] <= 5))), True))
)

def rule_55_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 55
        rule_55(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
