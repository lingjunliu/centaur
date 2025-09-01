import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Rule that input tensor and product of tile factor must be smaller than threshold on any dimension AND total memory is also within threshold and smaller than max int, and overall number of elements are less than max int32 and tensor size in bytes smaller than maximum 32 bits signed integer  (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (If(v["arg1_ndim"] < v["arg2_length"], v["arg1_ndim"] - 1, v["arg2_length"] - 1) + 1), And(And(And(Select(v["arg1_shape"], i) * Select(v["arg2_values"], i) < 20000, Select(v["arg1_shape"], 0) * Select(v["arg2_values"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg2_values"], 1) < 2000000000), Select(v["arg1_shape"], 0) * Select(v["arg2_values"], 0) * Select(v["arg1_shape"], v["arg1_ndim"] - 1) * Select(v["arg2_values"], v["arg2_length"] - 1) < 2147483647), Select(v["arg1_shape"], 0) * Select(v["arg2_values"], 0) * Select(v["arg1_shape"], 1) < 2147483647)) for i in range(6)])) if n else
          And([Implies(i < (If(v["arg1_ndim"] < v["arg2_length"], v["arg1_ndim"] - 1, v["arg2_length"] - 1) + 1), And(And(And(Select(v["arg1_shape"], i) * Select(v["arg2_values"], i) < 20000, Select(v["arg1_shape"], 0) * Select(v["arg2_values"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg2_values"], 1) < 2000000000), Select(v["arg1_shape"], 0) * Select(v["arg2_values"], 0) * Select(v["arg1_shape"], v["arg1_ndim"] - 1) * Select(v["arg2_values"], v["arg2_length"] - 1) < 2147483647), Select(v["arg1_shape"], 0) * Select(v["arg2_values"], 0) * Select(v["arg1_shape"], 1) < 2147483647)) for i in range(6)]))
)

def rule_34_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 34
        rule_34(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
