import numpy as np
import torch
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# For tf.image.extract_patches with VALID padding, the input spatial dimensions
# must be large enough to fit the effective filter size.
# effective_filter_size = (ksize - 1) * rate + 1
# input_size >= effective_filter_size must hold for both H and W.
# Error example: input [1,1,1,1], ksizes=[1,2,2,1], rates=[1,2,2,1]:
#   effective_filter_size = (2-1)*2+1 = 3 > 1 → ValueError

rule_1 = lambda s, v, n=False: (
    s.add(Not(And(
        v["arg1_h"] >= (v["ksize_h"] - 1) * v["rate_h"] + 1,
        v["arg1_w"] >= (v["ksize_w"] - 1) * v["rate_w"] + 1
    ))) if n else
    s.add(And(
        v["arg1_h"] >= (v["ksize_h"] - 1) * v["rate_h"] + 1,
        v["arg1_w"] >= (v["ksize_w"] - 1) * v["rate_w"] + 1
    ))
)

def rule_1_func(arg1, arg2, arg3, solver=None, neg=False):
    # arg1: images (np.ndarray, shape [N, H, W, C])
    # arg2: ksizes (e.g. [1, kH, kW, 1])
    # arg3: rates (e.g. [1, rH, rW, 1])
    # arg4: strides (not used in this rule, included for consistency)
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if arg1.ndim != 4:
            return False

        solver = Solver()
        arg1_h = Int('arg1_h')
        arg1_w = Int('arg1_w')
        ksize_h = Int('ksize_h')
        ksize_w = Int('ksize_w')
        rate_h = Int('rate_h')
        rate_w = Int('rate_w')

        solver.add(arg1_h == arg1.shape[1])
        solver.add(arg1_w == arg1.shape[2])
        solver.add(ksize_h == arg2[1])
        solver.add(ksize_w == arg2[2])
        solver.add(rate_h == arg3[1])
        solver.add(rate_w == arg3[2])

        rule_1(solver, {
            'arg1_h': arg1_h,
            'arg1_w': arg1_w,
            'ksize_h': ksize_h,
            'ksize_w': ksize_w,
            'rate_h': rate_h,
            'rate_w': rate_w
        })
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1(solver, {
            'arg1_h': arg1['arg1_h'],
            'arg1_w': arg1['arg1_w'],
            'ksize_h': arg1['ksize_h'],
            'ksize_w': arg1['ksize_w'],
            'rate_h': arg1['rate_h'],
            'rate_w': arg1['rate_w']
        }, neg)