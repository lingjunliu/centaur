
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedMul_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([4, 5, 6], dtype=np.int8)
    min_x = np.array(-1.0, dtype=np.float32)
    max_x = np.array(5.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(10.0, dtype=np.float32)

    input_dict = {
        "x": x.astype(np.qint8),
        "y": y.astype(np.qint8),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "quantized_mul_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(255.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(255.0, dtype=np.float32)
    input_dict = {
        "x": x.astype(np.quint8),
        "y": y.astype(np.quint8),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "quantized_mul_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6, 7, 8], dtype=np.int32)
    min_x = np.array(-100.0, dtype=np.float32)
    max_x = np.array(100.0, dtype=np.float32)
    min_y = np.array(-50.0, dtype=np.float32)
    max_y = np.array(50.0, dtype=np.float32)

    input_dict = {
        "x": x.astype(np.qint32),
        "y": y.astype(np.qint32),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "quantized_mul_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([4, 5, 6], dtype=np.int16)
    min_x = np.array(-1.0, dtype=np.float32)
    max_x = np.array(5.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(10.0, dtype=np.float32)

    input_dict = {
        "x": x.astype(np.qint16),
        "y": y.astype(np.qint16),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "quantized_mul_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint16)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(255.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(255.0, dtype=np.float32)
    input_dict = {
        "x": x.astype(np.quint16),
        "y": y.astype(np.quint16),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "quantized_mul_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1, 2, 3, 4], dtype=np.int8)
    y = np.array([5, 6, 7, 8], dtype=np.int8)
    min_x = np.array(-100.0, dtype=np.float32)
    max_x = np.array(100.0, dtype=np.float32)
    min_y = np.array(-50.0, dtype=np.float32)
    max_y = np.array(50.0, dtype=np.float32)

    input_dict = {
        "x": x.astype(np.qint8),
        "y": y.astype(np.qint8),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.quint8,
        "name": "quantized_mul_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1, 2], dtype=np.uint16)
    y = np.array([5, 6], dtype=np.uint16)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(65535.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(65535.0, dtype=np.float32)

    input_dict = {
        "x": x.astype(np.quint16),
        "y": y.astype(np.quint16),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint16,
        "name": "quantized_mul_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([-1, -2, -3], dtype=np.int8)
    y = np.array([4, 5, 6], dtype=np.int8)
    min_x = np.array(-5.0, dtype=np.float32)
    max_x = np.array(5.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(10.0, dtype=np.float32)

    input_dict = {
        "x": x.astype(np.qint8),
        "y": y.astype(np.qint8),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "quantized_mul_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([4, 5, 6], dtype=np.int8)
    min_x = np.array(-1.0, dtype=np.float32)
    max_x = np.array(5.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(10.0, dtype=np.float32)

    input_dict = {
        "x": x.astype(np.qint8),
        "y": y.astype(np.qint8),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint16,
        "name": "quantized_mul_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(255.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(255.0, dtype=np.float32)
    input_dict = {
        "x": x.astype(np.quint8),
        "y": y.astype(np.quint8),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.quint16,
        "name": "quantized_mul_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedMul"] = tf_raw_ops_QuantizedMul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedMul'.")

check_valid('tf.raw_ops.QuantizedMul', generated_inputs['tf.raw_ops.QuantizedMul'], lib="tf", suffix=0)
