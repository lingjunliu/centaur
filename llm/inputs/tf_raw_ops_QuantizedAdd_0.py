
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedAdd_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([4, 5, 6], dtype=np.int8)
    min_x = np.array(-1.0, dtype=np.float32)
    max_x = np.array(3.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(6.0, dtype=np.float32)

    x = tf.constant(x, dtype=tf.qint8)
    y = tf.constant(y, dtype=tf.qint8)
    min_x = tf.constant(min_x, dtype=tf.float32)
    max_x = tf.constant(max_x, dtype=tf.float32)
    min_y = tf.constant(min_y, dtype=tf.float32)
    max_y = tf.constant(max_y, dtype=tf.float32)


    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "add1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(4.0, dtype=np.float32)
    min_y = np.array(5.0, dtype=np.float32)
    max_y = np.array(9.0, dtype=np.float32)

    x = tf.constant(x, dtype=tf.quint8)
    y = tf.constant(y, dtype=tf.quint8)
    min_x = tf.constant(min_x, dtype=tf.float32)
    max_x = tf.constant(max_x, dtype=tf.float32)
    min_y = tf.constant(min_y, dtype=tf.float32)
    max_y = tf.constant(max_y, dtype=tf.float32)

    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "add2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6, 7, 8], dtype=np.int32)
    min_x = np.array(-10.0, dtype=np.float32)
    max_x = np.array(10.0, dtype=np.float32)
    min_y = np.array(-5.0, dtype=np.float32)
    max_y = np.array(15.0, dtype=np.float32)

    x = tf.constant(x, dtype=tf.qint32)
    y = tf.constant(y, dtype=tf.qint32)
    min_x = tf.constant(min_x, dtype=tf.float32)
    max_x = tf.constant(max_x, dtype=tf.float32)
    min_y = tf.constant(min_y, dtype=tf.float32)
    max_y = tf.constant(max_y, dtype=tf.float32)

    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "add3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    y = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int16)
    min_x = np.array(-2.0, dtype=np.float32)
    max_x = np.array(6.0, dtype=np.float32)
    min_y = np.array(6.0, dtype=np.float32)
    max_y = np.array(12.0, dtype=np.float32)

    x = tf.constant(x, dtype=tf.qint16)
    y = tf.constant(y, dtype=tf.qint16)
    min_x = tf.constant(min_x, dtype=tf.float32)
    max_x = tf.constant(max_x, dtype=tf.float32)
    min_y = tf.constant(min_y, dtype=tf.float32)
    max_y = tf.constant(max_y, dtype=tf.float32)

    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "add4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint16)
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.uint16)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(8.0, dtype=np.float32)
    min_y = np.array(8.0, dtype=np.float32)
    max_y = np.array(16.0, dtype=np.float32)

    x = tf.constant(x, dtype=tf.quint16)
    y = tf.constant(y, dtype=tf.quint16)
    min_x = tf.constant(min_x, dtype=tf.float32)
    max_x = tf.constant(max_x, dtype=tf.float32)
    min_y = tf.constant(min_y, dtype=tf.float32)
    max_y = tf.constant(max_y, dtype=tf.float32)


    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "add5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([-1, 0, 1], dtype=np.int8)
    y = np.array([2, -3, 4], dtype=np.int8)
    min_x = np.array(-2.0, dtype=np.float32)
    max_x = np.array(2.0, dtype=np.float32)
    min_y = np.array(-4.0, dtype=np.float32)
    max_y = np.array(4.0, dtype=np.float32)

    x = tf.constant(x, dtype=tf.qint8)
    y = tf.constant(y, dtype=tf.qint8)
    min_x = tf.constant(min_x, dtype=tf.float32)
    max_x = tf.constant(max_x, dtype=tf.float32)
    min_y = tf.constant(min_y, dtype=tf.float32)
    max_y = tf.constant(max_y, dtype=tf.float32)

    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint32,
        "name": "add6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([4, 5, 6], dtype=np.uint8)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(3.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(6.0, dtype=np.float32)

    x = tf.constant(x, dtype=tf.quint8)
    y = tf.constant(y, dtype=tf.quint8)
    min_x = tf.constant(min_x, dtype=tf.float32)
    max_x = tf.constant(max_x, dtype=tf.float32)
    min_y = tf.constant(min_y, dtype=tf.float32)
    max_y = tf.constant(max_y, dtype=tf.float32)

    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint16,
        "name": "add7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[5, 6], [7, 8]], dtype=np.int32)
    min_x = np.array(-10.0, dtype=np.float32)
    max_x = np.array(10.0, dtype=np.float32)
    min_y = np.array(-5.0, dtype=np.float32)
    max_y = np.array(15.0, dtype=np.float32)

    x = tf.constant(x, dtype=tf.qint32)
    y = tf.constant(y, dtype=tf.qint32)
    min_x = tf.constant(min_x, dtype=tf.float32)
    max_x = tf.constant(max_x, dtype=tf.float32)
    min_y = tf.constant(min_y, dtype=tf.float32)
    max_y = tf.constant(max_y, dtype=tf.float32)


    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.quint8,
        "name": "add8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([4, 5, 6], dtype=np.int8)
    min_x = np.array(-1.0, dtype=np.float32)
    max_x = np.array(3.0, dtype=np.float32)
    min_y = np.array(0.0, dtype=np.float32)
    max_y = np.array(6.0, dtype=np.float32)

    x = tf.constant(x, dtype=tf.qint8)
    y = tf.constant(y, dtype=tf.qint8)
    min_x = tf.constant(min_x, dtype=tf.float32)
    max_x = tf.constant(max_x, dtype=tf.float32)
    min_y = tf.constant(min_y, dtype=tf.float32)
    max_y = tf.constant(max_y, dtype=tf.float32)

    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.quint16,
        "name": "add9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint16)
    min_x = np.array(0.0, dtype=np.float32)
    max_x = np.array(4.0, dtype=np.float32)
    min_y = np.array(5.0, dtype=np.float32)
    max_y = np.array(9.0, dtype=np.float32)

    x = tf.constant(x, dtype=tf.quint16)
    y = tf.constant(y, dtype=tf.quint16)
    min_x = tf.constant(min_x, dtype=tf.float32)
    max_x = tf.constant(max_x, dtype=tf.float32)
    min_y = tf.constant(min_y, dtype=tf.float32)
    max_y = tf.constant(max_y, dtype=tf.float32)

    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": tf.qint8,
        "name": "add10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedAdd"] = tf_raw_ops_QuantizedAdd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedAdd'.")

check_valid('tf.raw_ops.QuantizedAdd', generated_inputs['tf.raw_ops.QuantizedAdd'], lib="tf", suffix=0)
