
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fill_inputs():
    list_of_inputs = []

    # Input 1: Basic integer fill
    dims = np.array([2, 3], dtype=np.int32)
    value = np.array(5, dtype=np.int32)
    input_dict = {"dims": dims, "value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape, integer value
    dims = np.array([1, 5, 2], dtype=np.int32)
    value = np.array(-2, dtype=np.int32)
    input_dict = {"dims": dims, "value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Long type
    dims = np.array([4, 1], dtype=np.int64)
    value = np.array(100, dtype=np.int32)
    input_dict = {"dims": dims, "value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  float value
    dims = np.array([2, 2, 2], dtype=np.int32)
    value = np.array(3.14, dtype=np.float32)
    input_dict = {"dims": dims, "value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  bool value
    dims = np.array([3, 3], dtype=np.int32)
    value = np.array(True, dtype=np.bool_)
    input_dict = {"dims": dims, "value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  float16 value
    dims = np.array([4, 4], dtype=np.int32)
    value = np.array(1.5, dtype=np.float16)
    input_dict = {"dims": dims, "value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  int8 value
    dims = np.array([5, 1], dtype=np.int32)
    value = np.array(-10, dtype=np.int8)
    input_dict = {"dims": dims, "value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  uint8 value
    dims = np.array([1, 5], dtype=np.int32)
    value = np.array(200, dtype=np.uint8)
    input_dict = {"dims": dims, "value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64 dims, int64 value
    dims = np.array([2, 3], dtype=np.int64)
    value = np.array(1234567890, dtype=np.int64)
    input_dict = {"dims": dims, "value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty dims, float32 value
    dims = np.array([], dtype=np.int32)
    value = np.array(2.5, dtype=np.float32)
    input_dict = {"dims": dims, "value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Fill"] = tf_raw_ops_fill_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Fill' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Fill'.")

check_valid('tf.raw_ops.Fill', generated_inputs['tf.raw_ops.Fill'], lib="tf", suffix=0)
