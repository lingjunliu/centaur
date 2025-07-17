
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dequantize_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([0, 1, 2, 3], dtype=np.int8)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(6.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    narrow_range_bool = False
    axis_int = -1
    dtype_tf = tf.float32
    name_str = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "dtype": dtype_tf,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([-128, 0, 127], dtype=np.qint8)
    min_range_tensor = np.array(-10.0, dtype=np.float32)
    max_range_tensor = np.array(10.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    narrow_range_bool = False
    axis_int = -1
    dtype_tf = tf.float32
    name_str = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "dtype": dtype_tf,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([0, 65535], dtype=np.int16)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(1.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    narrow_range_bool = True
    axis_int = -1
    dtype_tf = tf.float32
    name_str = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "dtype": dtype_tf,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([-32768, 32767], dtype=np.qint16)
    min_range_tensor = np.array(-1.0, dtype=np.float32)
    max_range_tensor = np.array(1.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    narrow_range_bool = True
    axis_int = -1
    dtype_tf = tf.float32
    name_str = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "dtype": dtype_tf,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([0, 1000000], dtype=np.int32)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(100.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    narrow_range_bool = False
    axis_int = -1
    dtype_tf = tf.float32
    name_str = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "dtype": dtype_tf,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: MIN_FIRST mode
    input_tensor = np.array([0, 255], dtype=np.uint8)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(5.0, dtype=np.float32)
    mode_str = "MIN_FIRST"
    narrow_range_bool = False
    axis_int = -1
    dtype_tf = tf.float32
    name_str = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "dtype": dtype_tf,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: SCALED mode
    input_tensor = np.array([0, 255], dtype=np.uint8)
    min_range_tensor = np.array(-1.0, dtype=np.float32)
    max_range_tensor = np.array(1.0, dtype=np.float32)
    mode_str = "SCALED"
    narrow_range_bool = False
    axis_int = -1
    dtype_tf = tf.float32
    name_str = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "dtype": dtype_tf,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    input_tensor = np.array([0, 1, 2, 3], dtype=np.uint8)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(6.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    narrow_range_bool = False
    axis_int = -1
    dtype_tf = tf.bfloat16
    name_str = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "dtype": dtype_tf,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.uint8)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(10.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    narrow_range_bool = False
    axis_int = -1
    dtype_tf = tf.float32
    name_str = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "dtype": dtype_tf,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([0, 1, 2, 3], dtype=np.uint8)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(6.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    narrow_range_bool = True
    axis_int = -1
    dtype_tf = tf.float32
    name_str = None

    input_dict = {
        "input": input_tensor,
        "min_range": min_range_tensor,
        "max_range": max_range_tensor,
        "mode": mode_str,
        "narrow_range": narrow_range_bool,
        "axis": axis_int,
        "dtype": dtype_tf,
        "name": name_str
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
