
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_clipbyvalue_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar min/max
    t = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    clip_value_min = np.array(0.0, dtype=np.float32)
    clip_value_max = np.array(1.0, dtype=np.float32)
    input_dict = {"t": t, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ClipByValue"] = tf_raw_ops_clipbyvalue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ClipByValue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ClipByValue'.")

check_valid('tf.raw_ops.ClipByValue', generated_inputs['tf.raw_ops.ClipByValue'], lib="tf", suffix=0)
