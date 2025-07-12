
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_dequantize_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([0, 1, 2, 3], dtype=np.int16)
    min_range = np.float32(0.0)
    max_range = np.float32(6.0)
    mode = "MIN_COMBINED"
    name = "dequantize_example_1"
    axis = None
    narrow_range = False
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint16),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([-127, 0, 127], dtype=np.int8)
    min_range = np.float32(-10.0)
    max_range = np.float32(10.0)
    mode = "MIN_COMBINED"
    name = "dequantize_example_2"
    axis = None
    narrow_range = False
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint8),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([0, 127], dtype=np.int8)
    min_range = np.float32(0.0)
    max_range = np.float32(127.0)
    mode = "MIN_FIRST"
    name = "dequantize_example_3"
    axis = None
    narrow_range = False
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint8),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([-127, 127], dtype=np.int16)
    min_range = np.float32(-1.0)
    max_range = np.float32(1.0)
    mode = "MIN_FIRST"
    name = "dequantize_example_4"
    axis = None
    narrow_range = True
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint16),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([0, 32767], dtype=np.int16)
    min_range = np.float32(0.0)
    max_range = np.float32(100.0)
    mode = "SCALED"
    name = "dequantize_example_5"
    axis = None
    narrow_range = False
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint16),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([-32767, 32767], dtype=np.int16)
    min_range = np.float32(-50.0)
    max_range = np.float32(50.0)
    mode = "SCALED"
    name = "dequantize_example_6"
    axis = None
    narrow_range = True
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint16),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_tensor = np.array([-127, 127], dtype=np.int8)
    min_range = np.float32(-1000.0)
    max_range = np.float32(1000.0)
    mode = "MIN_COMBINED"
    name = "dequantize_example_7"
    axis = None
    narrow_range = False
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint8),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int16)
    min_range = np.float32(0.0)
    max_range = np.float32(1.0)
    mode = "MIN_COMBINED"
    name = "dequantize_example_8"
    axis = None
    narrow_range = False
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint16),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([0, 1, 2, 3], dtype=np.int16)
    min_range = np.float32(0.0)
    max_range = np.float32(6.0)
    mode = "MIN_COMBINED"
    name = "dequantize_example_9"
    axis = 0
    narrow_range = False
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint16),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([-127, 0, 127], dtype=np.int8)
    min_range = np.float32(-10.0)
    max_range = np.float32(10.0)
    mode = "MIN_COMBINED"
    name = "dequantize_example_10"
    axis = 0
    narrow_range = False
    dtype = tf.bfloat16

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint8),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array([0, 1, 2, 3], dtype=np.int16)
    min_range = np.float32(0.0)
    max_range = np.float32(6.0)
    mode = "SCALED"
    name = "dequantize_example_11"
    axis = None
    narrow_range = True
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint16),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input_tensor = np.array([0, 1, 2, 3], dtype=np.int32)
    min_range = np.float32(0.0)
    max_range = np.float32(6.0)
    mode = "SCALED"
    name = "dequantize_example_12"
    axis = None
    narrow_range = True
    dtype = tf.float32

    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.qint32),
        "min_range": tf.constant(min_range, dtype=tf.float32),
        "max_range": tf.constant(max_range, dtype=tf.float32),
        "mode": mode,
        "name": name,
        "axis": axis,
        "narrow_range": narrow_range,
        "dtype": dtype
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
