import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# inputs must be empty (Rule 110)

rule_110 = lambda s, v, n=False: (
    s.add(Not(v["arg1_length"] == 0) if n else
          v["arg1_length"] == 0)
)