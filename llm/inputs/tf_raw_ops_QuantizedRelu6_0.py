
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedRelu6_inputs():
    list_of_inputs = []

    # Input 1
    features = np.array([-1, 0, 1, 2, 6, 7], dtype=np.int8)
    min_features = np.array([-2.0], dtype=np.float32)
    max_features = np.array([8.0], dtype=np.float32)
    out_type = tf.quint8
    name = "relu6_1"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.qint8),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    features = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([6.0], dtype=np.float32)
    out_type = tf.quint8
    name = "relu6_2"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.quint8),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    features = np.array([-5, -2, 0, 3, 7, 10], dtype=np.int32)
    min_features = np.array([-10.0], dtype=np.float32)
    max_features = np.array([10.0], dtype=np.float32)
    out_type = tf.qint32
    name = "relu6_3"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.qint32),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    features = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([5.0], dtype=np.float32)
    out_type = tf.qint16
    name = "relu6_4"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.qint16),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    features = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([7.0], dtype=np.float32)
    out_type = tf.quint16
    name = "relu6_5"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.quint16),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    features = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int8)
    min_features = np.array([-1.0], dtype=np.float32)
    max_features = np.array([4.0], dtype=np.float32)
    out_type = tf.qint8
    name = "relu6_6"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.qint8),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    features = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    min_features = np.array([1.0], dtype=np.float32)
    max_features = np.array([8.0], dtype=np.float32)
    out_type = tf.quint8
    name = "relu6_7"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.quint8),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    features = np.array([[-2, -1, 0], [1, 2, 3]], dtype=np.int32)
    min_features = np.array([-3.0], dtype=np.float32)
    max_features = np.array([3.0], dtype=np.float32)
    out_type = tf.qint32
    name = "relu6_8"
    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.qint32),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    features = np.array([[-100, 0, 100], [1000, 2000, 3000]], dtype=np.int16)
    min_features = np.array([-200.0], dtype=np.float32)
    max_features = np.array([4000.0], dtype=np.float32)
    out_type = tf.qint16
    name = "relu6_9"
    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.qint16),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    features = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([6.0], dtype=np.float32)
    out_type = tf.quint16
    name = "relu6_10"
    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.quint16),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedRelu6"] = tf_raw_ops_QuantizedRelu6_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedRelu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedRelu6'.")

check_valid('tf.raw_ops.QuantizedRelu6', generated_inputs['tf.raw_ops.QuantizedRelu6'], lib="tf", suffix=0)
