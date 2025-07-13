
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_range_inputs():
    list_of_inputs = []

    # Input 1: Basic positive range
    start = np.array(3, dtype=np.int32)
    limit = np.array(18, dtype=np.int32)
    delta = np.array(3, dtype=np.int32)
    name = "range_1"
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative delta
    start = np.array(10, dtype=np.int32)
    limit = np.array(0, dtype=np.int32)
    delta = np.array(-2, dtype=np.int32)
    name = "range_2"
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Start greater than limit, positive delta
    start = np.array(5, dtype=np.int32)
    limit = np.array(2, dtype=np.int32)
    delta = np.array(1, dtype=np.int32)
    name = "range_3"
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float type
    start = np.array(1.0, dtype=np.float32)
    limit = np.array(5.0, dtype=np.float32)
    delta = np.array(0.5, dtype=np.float32)
    name = "range_5"
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative start, limit, and delta
    start = np.array(-10, dtype=np.int32)
    limit = np.array(-1, dtype=np.int32)
    delta = np.array(1, dtype=np.int32)
    name = "range_6"
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Range"] = tf_raw_ops_range_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Range' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Range'.")

check_valid('tf.raw_ops.Range', generated_inputs['tf.raw_ops.Range'], lib="tf", suffix=0)
