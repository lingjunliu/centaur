
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dequantize_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([10, 20, 30], dtype=np.int8)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(100.0, dtype=np.float32)
    mode = "MIN_COMBINED"
    narrow_range = False
    axis = -1
    dtype = tf.float32
    name = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    min_range_tensor = np.array(-1.0, dtype=np.float32)
    max_range_tensor = np.array(1.0, dtype=np.float32)
    mode = "MIN_FIRST"
    narrow_range = True
    axis = 0
    dtype = tf.float32
    name = "dequantize_example"

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int8)
    min_range_tensor = np.array(-5.0, dtype=np.float32)
    max_range_tensor = np.array(5.0, dtype=np.float32)
    mode = "SCALED"
    narrow_range = False
    axis = 1
    dtype = tf.float32
    name = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1000, 2000, 3000], dtype=np.int32)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(10000.0, dtype=np.float32)
    mode = "MIN_COMBINED"
    narrow_range = True
    axis = -1
    dtype = tf.float32
    name = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[10, 20], [30, 40]], dtype=np.int16)
    min_range_tensor = np.array(-10.0, dtype=np.float32)
    max_range_tensor = np.array(10.0, dtype=np.float32)
    mode = "MIN_FIRST"
    narrow_range = False
    axis = 0
    dtype = tf.float32
    name = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([10, 20, 30], dtype=np.uint16)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(50.0, dtype=np.float32)
    mode = "SCALED"
    narrow_range = True
    axis = -1
    dtype = tf.float32
    name = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([10, 20, 30], dtype=np.int8)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(100.0, dtype=np.float32)
    mode = "MIN_COMBINED"
    narrow_range = False
    axis = -1
    dtype = tf.bfloat16
    name = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    min_range_tensor = np.array(-100.0, dtype=np.float32)
    max_range_tensor = np.array(100.0, dtype=np.float32)
    mode = "MIN_COMBINED"
    narrow_range = False
    axis = 0
    dtype = tf.float32
    name = "test_dequantize"

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    min_range_tensor = np.array(5.0, dtype=np.float32)
    max_range_tensor = np.array(10.0, dtype=np.float32)
    mode = "MIN_FIRST"
    narrow_range = True
    axis = -1
    dtype = tf.float32
    name = "test_dequantize"

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    min_range_tensor = np.array(-5.0, dtype=np.float32)
    max_range_tensor = np.array(10.0, dtype=np.float32)
    mode = "SCALED"
    narrow_range = True
    axis = -1
    dtype = tf.float32
    name = "test_dequantize"

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode,
        "narrow_range": narrow_range,
        "axis": axis,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Dequantize"] = tf_raw_ops_dequantize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Dequantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dequantize'.")

check_valid('tf.raw_ops.Dequantize', generated_inputs['tf.raw_ops.Dequantize'], lib="tf", suffix=0)
