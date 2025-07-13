
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fill_inputs():
    list_of_inputs = []

    # Input 1: Basic example with int32 dims and float value
    dims = np.array([2, 3], dtype=np.int32)
    value = np.array(9.0, dtype=np.float32)
    input_dict = {"dims": tf.convert_to_tensor(dims, dtype=tf.int32), "value": tf.convert_to_tensor(value, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int64 dims and int value
    dims = np.array([5], dtype=np.int64)
    value = np.array(-3, dtype=np.int32)
    input_dict = {"dims": tf.convert_to_tensor(dims, dtype=tf.int64), "value": tf.convert_to_tensor(value, dtype=tf.int32), "name": "fill_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional with bool value
    dims = np.array([2, 2, 2], dtype=np.int32)
    value = np.array(True, dtype=np.bool_)
    input_dict = {"dims": tf.convert_to_tensor(dims, dtype=tf.int32), "value": tf.convert_to_tensor(value, dtype=tf.bool), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger dimensions
    dims = np.array([10, 5], dtype=np.int64)
    value = np.array(1, dtype=np.int64)
    input_dict = {"dims": tf.convert_to_tensor(dims, dtype=tf.int64), "value": tf.convert_to_tensor(value, dtype=tf.int64), "name": "fill_example_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element dimension
    dims = np.array([1], dtype=np.int32)
    value = np.array(3.14, dtype=np.float64)
    input_dict = {"dims": tf.convert_to_tensor(dims, dtype=tf.int32), "value": tf.convert_to_tensor(value, dtype=tf.float64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int8 value
    dims = np.array([3, 4], dtype=np.int64)
    value = np.array(-5, dtype=np.int8)
    input_dict = {"dims": tf.convert_to_tensor(dims, dtype=tf.int64), "value": tf.convert_to_tensor(value, dtype=tf.int8), "name": "fill_example_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8 value
    dims = np.array([2, 5], dtype=np.int32)
    value = np.array(255, dtype=np.uint8)
    input_dict = {"dims": tf.convert_to_tensor(dims, dtype=tf.int32), "value": tf.convert_to_tensor(value, dtype=tf.uint8), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty array dimension
    dims = np.array([0], dtype=np.int64)
    value = np.array(10, dtype=np.int32)
    input_dict = {"dims": tf.convert_to_tensor(dims, dtype=tf.int64), "value": tf.convert_to_tensor(value, dtype=tf.int32), "name": "fill_example_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger dimension sizes.
    dims = np.array([100, 100], dtype=np.int32)
    value = np.array(0.5, dtype=np.float32)
    input_dict = {"dims": tf.convert_to_tensor(dims, dtype=tf.int32), "value": tf.convert_to_tensor(value, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16 value
    dims = np.array([4, 2], dtype=np.int64)
    value = np.array(0.123, dtype=np.float16)
    input_dict = {"dims": tf.convert_to_tensor(dims, dtype=tf.int64), "value": tf.convert_to_tensor(value, dtype=tf.float16), "name": "fill_example_10"}
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
