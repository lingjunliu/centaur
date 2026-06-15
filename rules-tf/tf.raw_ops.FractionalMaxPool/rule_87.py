import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# pooling ratio must not be higher than input dimension size (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_length"] == 4, And(Select(v["arg2_values"], 1) <= Select(v["arg1_shape"], 1), Select(v["arg2_values"], 2) <= Select(v["arg1_shape"], 2)), True)) if n else
          If(v["arg2_length"] == 4, And(Select(v["arg2_values"], 1) <= Select(v["arg1_shape"], 1), Select(v["arg2_values"], 2) <= Select(v["arg1_shape"], 2)), True))
)

def rule_87_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, list) and all(isinstance(e, (float, np.floating)) for e in arg2)) or (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2))):
            return False
