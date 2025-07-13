
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedRelu_inputs():
    list_of_inputs = []

    # Input 1
    features = np.array([0, 0, 1, 2], dtype=np.int8)
    min_features = np.array([-2.0], dtype=np.float32)
    max_features = np.array([2.0], dtype=np.float32)
    out_type = tf.quint8
    name = "relu_1"

    input_dict = {
        "features": features,
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    features = np.array([[0, 0], [1, 2]], dtype=np.uint8)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([3.0], dtype=np.float32)
    out_type = tf.quint8
    name = "relu_2"

    input_dict = {
        "features": features,
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    features = np.array([0, 0, 10, 20], dtype=np.int32)
    min_features = np.array([-20.0], dtype=np.float32)
    max_features = np.array([20.0], dtype=np.float32)
    out_type = tf.qint32
    name = "relu_3"

    input_dict = {
        "features": features,
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    features = np.array([[0, 0], [1, 2]], dtype=np.int16)
    min_features = np.array([-2.0], dtype=np.float32)
    max_features = np.array([2.0], dtype=np.float32)
    out_type = tf.qint16
    name = "relu_4"

    input_dict = {
        "features": features,
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    features = np.array([0, 0, 100, 200], dtype=np.uint16)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([300.0], dtype=np.float32)
    out_type = tf.quint16
    name = "relu_5"

    input_dict = {
        "features": features,
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    features = np.array([[0, 0, 1], [2, 0, 4]], dtype=np.int8)
    min_features = np.array([-4.0], dtype=np.float32)
    max_features = np.array([4.0], dtype=np.float32)
    out_type = tf.qint8
    name = "relu_6"

    input_dict = {
        "features": features,
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    features = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    min_features = np.array([1.0], dtype=np.float32)
    max_features = np.array([8.0], dtype=np.float32)
    out_type = tf.quint8
    name = "relu_7"

    input_dict = {
        "features": features,
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    features = np.array([[0, 0, 1], [2, 0, 4]], dtype=np.int32)
    min_features = np.array([-10.0], dtype=np.float32)
    max_features = np.array([10.0], dtype=np.float32)
    out_type = tf.qint32
    name = "relu_8"

    input_dict = {
        "features": features,
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    features = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    min_features = np.array([-5.0], dtype=np.float32)
    max_features = np.array([10.0], dtype=np.float32)
    out_type = tf.qint16
    name = "relu_9"

    input_dict = {
        "features": features,
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    features = np.array([[0, 0, 10], [20, 0, 40]], dtype=np.uint16)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([50.0], dtype=np.float32)
    out_type = tf.quint16
    name = "relu_10"

    input_dict = {
        "features": features,
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedRelu"] = tf_raw_ops_QuantizedRelu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedRelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedRelu'.")

check_valid('tf.raw_ops.QuantizedRelu', generated_inputs['tf.raw_ops.QuantizedRelu'], lib="tf", suffix=0)
