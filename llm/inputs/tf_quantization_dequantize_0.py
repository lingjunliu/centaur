
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_dequantize_inputs():
    list_of_inputs = []

    # Input 1: Basic case with quint8
    input_tensor = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(6.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "MIN_COMBINED",
        "name": None,
        "axis": -1,
        "narrow_range": False,
        "dtype": tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: qint8 with negative range
    input_tensor = np.array([-128, -64, 0, 64, 127], dtype=np.int8)
    min_range_tensor = np.float32(-5.0)
    max_range_tensor = np.float32(5.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "MIN_COMBINED",
        "name": None,
        "axis": -1,
        "narrow_range": False,
        "dtype": tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: quint16 with different range
    input_tensor = np.array([0, 16384, 32767, 49151, 65535], dtype=np.uint16)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(10.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "MIN_COMBINED",
        "name": "dequantize_example",
        "axis": -1,
        "narrow_range": False,
        "dtype": tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: qint32
    input_tensor = np.array([-2147483648, -1073741824, 0, 1073741823, 2147483647], dtype=np.int32)
    min_range_tensor = np.float32(-1.0)
    max_range_tensor = np.float32(1.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "MIN_COMBINED",
        "name": None,
        "axis": -1,
        "narrow_range": False,
        "dtype": tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: MIN_FIRST mode
    input_tensor = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(6.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "MIN_FIRST",
        "name": None,
        "axis": -1,
        "narrow_range": False,
        "dtype": tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: SCALED mode
    input_tensor = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(6.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "SCALED",
        "name": None,
        "axis": -1,
        "narrow_range": False,
        "dtype": tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: SCALED mode with narrow_range=True
    input_tensor = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(6.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "SCALED",
        "name": None,
        "axis": -1,
        "narrow_range": True,
        "dtype": tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: qint16 with negative range, narrow_range = True
    input_tensor = np.array([-32768, -16384, 0, 16383, 32767], dtype=np.int16)
    min_range_tensor = np.float32(-5.0)
    max_range_tensor = np.float32(5.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "MIN_COMBINED",
        "name": None,
        "axis": -1,
        "narrow_range": True,
        "dtype": tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: qint8 2D Tensor
    input_tensor = np.array([[-128, -64], [0, 64], [127, -10]], dtype=np.int8)
    min_range_tensor = np.float32(-5.0)
    max_range_tensor = np.float32(5.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "MIN_COMBINED",
        "name": None,
        "axis": -1,
        "narrow_range": False,
        "dtype": tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: quint8 3D Tensor
    input_tensor = np.array([[[0, 64], [128, 192]], [[255, 0], [64, 128]]], dtype=np.uint8)
    min_range_tensor = np.float32(0.0)
    max_range_tensor = np.float32(6.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "MIN_COMBINED",
        "name": None,
        "axis": -1,
        "narrow_range": False,
        "dtype": tf.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: qint16 1D Tensor
    input_tensor = np.array([-32768, -16384, 0, 16383, 32767], dtype=np.int16)
    min_range_tensor = np.float32(-10.0)
    max_range_tensor = np.float32(10.0)
    input_dict = {
        "input": tf.constant(input_tensor, dtype=tf.int32),
        "min_range": tf.constant(min_range_tensor),
        "max_range": tf.constant(max_range_tensor),
        "mode": "MIN_COMBINED",
        "name": None,
        "axis": None,
        "narrow_range": False,
        "dtype": tf.float32
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
