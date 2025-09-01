import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# shared_embedding_collection_name should be different from ckpt_to_load_from if shared_embedding_collection_name is specified (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] != 6, v["arg1_value"] != v["arg2_value"], True)) if n else
          If(v["arg1_value"] != 6, v["arg1_value"] != v["arg2_value"], True))
)

def rule_69_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 69
        rule_69(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
