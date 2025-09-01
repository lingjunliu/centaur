import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# boxes must have shape [num_boxes, 4] and scores have shape [num_boxes], max_output_size < number of boxes, score_threshold scalar, iou_threshold scalar (Rule 140)

rule_140 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(Select(v["arg1_shape"], 1) == 4, v["arg2_ndim"] == 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), Select(v["arg1_shape"], 0) > v["arg3_value"]), v["arg4_ndim"] == 0), v["arg5_ndim"] == 0)) if n else
          And(And(And(And(And(Select(v["arg1_shape"], 1) == 4, v["arg2_ndim"] == 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), Select(v["arg1_shape"], 0) > v["arg3_value"]), v["arg4_ndim"] == 0), v["arg5_ndim"] == 0))
)

def rule_140_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg5_ndim = Int('arg5_ndim')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg5_ndim == arg5.ndim)

        # Constraints for rule 140
        rule_140(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_140(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim'], 'arg5_ndim': arg5['ndim']}, neg)
