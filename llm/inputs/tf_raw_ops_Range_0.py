
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_range_inputs():
    list_of_inputs = []

    # Input 1: int32, positive values
    start = np.array(3, dtype=np.int32)
    limit = np.array(18, dtype=np.int32)
    delta = np.array(3, dtype=np.int32)
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": "range_int32_pos"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int64, negative values
    start = np.array(-10, dtype=np.int64)
    limit = np.array(5, dtype=np.int64)
    delta = np.array(2, dtype=np.int64)
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": "range_int64_neg"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, fractional values
    start = np.array(1.5, dtype=np.float32)
    limit = np.array(5.5, dtype=np.float32)
    delta = np.array(0.5, dtype=np.float32)
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": "range_float32_frac"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, negative fractional values
    start = np.array(-2.7, dtype=np.float64)
    limit = np.array(1.3, dtype=np.float64)
    delta = np.array(0.8, dtype=np.float64)
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": "range_float64_neg_frac"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32, start > limit, positive delta
    start = np.array(10, dtype=np.int32)
    limit = np.array(1, dtype=np.int32)
    delta = np.array(1, dtype=np.int32)
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": "range_int32_start_greater"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32, start < limit, negative delta
    start = np.array(1, dtype=np.int32)
    limit = np.array(10, dtype=np.int32)
    delta = np.array(-1, dtype=np.int32)
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": "range_int32_negative_delta"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16, positive values
    start = np.array(2.0, dtype=np.float16) # bfloat16 unavailable in numpy
    limit = np.array(10.0, dtype=np.float16)
    delta = np.array(1.5, dtype=np.float16)
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": "range_bfloat16_pos"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: int32, positive values
    start = np.array(0, dtype=np.int32)
    limit = np.array(100, dtype=np.int32)
    delta = np.array(25, dtype=np.int32)
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": "range_int32_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64, positive values
    start = np.array(0, dtype=np.int64)
    limit = np.array(100, dtype=np.int64)
    delta = np.array(25, dtype=np.int64)
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": "range_int64_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, positive values
    start = np.array(0.0, dtype=np.float32)
    limit = np.array(100.0, dtype=np.float32)
    delta = np.array(25.0, dtype=np.float32)
    input_dict = {"start": start, "limit": limit, "delta": delta, "name": "range_float32_positive"}
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
