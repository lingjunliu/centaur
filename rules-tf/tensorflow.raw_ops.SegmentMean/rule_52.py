import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Segment_ids cannot be uint8, qint8, quint8, qint32, bfloat16, qint16, quint16, uint16, half, uint32, uint64 type (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_dtype"] != 5, v["arg1_dtype"] != 13), v["arg1_dtype"] != 14), v["arg1_dtype"] != 15), v["arg1_dtype"] != 16), v["arg1_dtype"] != 17), v["arg1_dtype"] != 18), v["arg1_dtype"] != 6)) if n else
          And(And(And(And(And(And(And(v["arg1_dtype"] != 5, v["arg1_dtype"] != 13), v["arg1_dtype"] != 14), v["arg1_dtype"] != 15), v["arg1_dtype"] != 16), v["arg1_dtype"] != 17), v["arg1_dtype"] != 18), v["arg1_dtype"] != 6))
)

def rule_52_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 52
        rule_52(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_dtype': arg1['dtype']}, neg)
