import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# real, imag has float32 if Tout is complex64 and real, imag has float64 if Tout is complex128 and only then  (Rule 132)

rule_132 = lambda s, v, n=False: (
    s.add(Not(And(((v["arg1_value"] == 10) == (And((v["arg2_dtype"] == 8), (v["arg3_dtype"] == 8)))), ((v["arg1_value"] == 11) == (And((v["arg2_dtype"] == 9), (v["arg3_dtype"] == 9)))))) if n else
          And(((v["arg1_value"] == 10) == (And((v["arg2_dtype"] == 8), (v["arg3_dtype"] == 8)))), ((v["arg1_value"] == 11) == (And((v["arg2_dtype"] == 9), (v["arg3_dtype"] == 9))))))
)

def rule_132_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 132
        rule_132(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_132(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype']}, neg)
