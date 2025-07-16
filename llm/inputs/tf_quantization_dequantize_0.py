
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_dequantize_inputs():
    list_of_inputs = []

    # Input 1
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(6.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_example_1"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        'input': tf.constant([0, 1, 2, 3], dtype=tf.qint8),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }
    list_of_inputs.append(input_dict)

    # Input 2
    min_range_tensor = np.array(-1.0, dtype=np.float32)
    max_range_tensor = np.array(1.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_example_2"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        'input': tf.constant([-128, 0, 127], dtype=tf.qint8),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }

    list_of_inputs.append(input_dict)

    # Input 3
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(10.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_example_3"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32
    input_dict = {
        'input': tf.constant([0, 255], dtype=tf.quint8),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }

    list_of_inputs.append(input_dict)

    # Input 4
    min_range_tensor = np.array(-5.0, dtype=np.float32)
    max_range_tensor = np.array(5.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_example_4"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        'input': tf.constant([-32768, 32767], dtype=tf.qint16),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }

    list_of_inputs.append(input_dict)

    # Input 5
    min_range_tensor = np.array(-100.0, dtype=np.float32)
    max_range_tensor = np.array(100.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_example_5"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        'input': tf.constant([-2147483648, 2147483647], dtype=tf.qint32),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }
    list_of_inputs.append(input_dict)

    # Input 6
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(6.0, dtype=np.float32)
    mode_str = "MIN_FIRST"
    name_str = "dequantize_example_6"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        'input': tf.constant([0, 1, 2, 3], dtype=tf.qint8),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }
    list_of_inputs.append(input_dict)

    # Input 7
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(6.0, dtype=np.float32)
    mode_str = "SCALED"
    name_str = "dequantize_example_7"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        'input': tf.constant([0, 1, 2, 3], dtype=tf.quint8),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }
    list_of_inputs.append(input_dict)

    # Input 8
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(6.0, dtype=np.float32)
    mode_str = "SCALED"
    name_str = "dequantize_example_8"
    axis_int = -1
    narrow_range_bool = True
    dtype_type = tf.float32

    input_dict = {
        'input': tf.constant([0, 1, 2, 3], dtype=tf.quint8),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }
    list_of_inputs.append(input_dict)

    # Input 9, multi-dimensional
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(6.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_example_9"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        'input': tf.constant([[0, 1], [2, 3]], dtype=tf.quint8),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }
    list_of_inputs.append(input_dict)

    # Input 10, bfloat16
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(6.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_example_10"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.bfloat16

    input_dict = {
        'input': tf.constant([0, 1, 2, 3], dtype=tf.quint8),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }
    list_of_inputs.append(input_dict)

     # Input 11, axis specified
    min_range_tensor = np.array(0.0, dtype=np.float32)
    max_range_tensor = np.array(6.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_example_11"
    axis_int = 0
    narrow_range_bool = False
    dtype_type = tf.float32

    input_dict = {
        'input': tf.constant([[0, 1], [2, 3]], dtype=tf.quint8),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }
    list_of_inputs.append(input_dict)

    # Input 12, qint16
    min_range_tensor = np.array(-10.0, dtype=np.float32)
    max_range_tensor = np.array(10.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_example_12"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32
    input_dict = {
        'input': tf.constant([0, 1000, -1000], dtype=tf.qint16),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }
    list_of_inputs.append(input_dict)

    input_tensor = np.array([0, 1000, -1000], dtype=np.int32)
    min_range_tensor = np.array(-10.0, dtype=np.float32)
    max_range_tensor = np.array(10.0, dtype=np.float32)
    mode_str = "MIN_COMBINED"
    name_str = "dequantize_example_13"
    axis_int = -1
    narrow_range_bool = False
    dtype_type = tf.float32
    input_dict = {
        'input': tf.constant([0, 1000, -1000], dtype=tf.qint32),
        'min_range': min_range_tensor,
        'max_range': max_range_tensor,
        'mode': mode_str,
        'name': name_str,
        'axis': axis_int,
        'narrow_range': narrow_range_bool,
        'dtype': dtype_type
    }
    list_of_inputs.append(input_dict)

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
