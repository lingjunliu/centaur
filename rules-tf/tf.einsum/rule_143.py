import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# number of inputs must match the einsum equation (Rule 143)

rule_143 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 2, v["arg2_length"] == 2, True)) if n else
          If(v["arg1_value"] == 2, v["arg2_length"] == 2, True))
)