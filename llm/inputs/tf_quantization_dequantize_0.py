
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_dequantize_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([10, 50, 100], dtype=np.int8)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(100.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_1"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint8),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([1, 2, 3, 4], dtype=np.uint8)
    min_range_tensor = np.array(-1.0, dtype=np.float32)
    max_range_tensor = np.array(1.0, dtype=np.float32)
    mode_str = "MIN_FIRST"
    name_str = "dequantize_2"
    axis_int = 0
    narrow_range_bool = True
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.quint8),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-1, 0], [1, 2]], dtype=np.int8)
    min_range_tensor = np.array(-5.0, dtype=np.float32)
    max_range_tensor = np.array(5.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_3"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint8),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1000, 2000, 3000], dtype=np.int32)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(10000.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_4"
    axis_int = None
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    min_range_tensor = np.array(-2.0, dtype=np.float32)
    max_range_tensor = np.array(2.0, dtype=np.float32)
    mode_str = "MIN_FIRST"
    name_str = "dequantize_5"
    axis_int = 1
    narrow_range_bool = True
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.quint8),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[-10, -5], [0, 5], [10, 15]], dtype=np.int16)
    min_range_tensor = np.array(-20.0, dtype=np.float32)
    max_range_tensor = np.array(20.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_6"
    axis_int = 0
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint16),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(5.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_7"
    axis_int = -1
    narrow_range_bool = True
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.quint16),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    input_tensor = np.array([-128, 0, 127], dtype=np.int8)
    min_range_tensor = np.array(-1.0, dtype=np.float32)
    max_range_tensor = np.array(1.0, dtype=np.float32)
    mode_str = "MIN_FIRST"
    name_str = "dequantize_8"
    axis_int = 0
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint8),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([0, 63, 127, 191, 255], dtype=np.uint8)
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(255.0, dtype=np.float32)
    mode_str = "SCALED"
    name_str = "dequantize_9"
    axis_int = -1
    narrow_range_bool = True
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.quint8),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    min_range_tensor = np.array(-10.0, dtype=np.float32)
    max_range_tensor = np.array(10.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_10"
    axis_int = 1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11, Remove bfloat16 to avoid errors
    input_tensor = np.array([1, 2, 3], dtype=np.int8)
    min_range_tensor = np.array(-1.0, dtype=np.float32)
    max_range_tensor = np.array(1.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_11"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint8),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": mode_str,
        "name": name_str,
        "axis": axis_int,
        "narrow_range": narrow_range_bool,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.quantization.dequantize"] = tf_quantization_dequantize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.quantization.dequantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.dequantize'.")

check_valid('tf.quantization.dequantize', generated_inputs['tf.quantization.dequantize'], lib="tf", suffix=0)
