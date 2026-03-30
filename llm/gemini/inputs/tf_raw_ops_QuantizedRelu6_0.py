
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_quantizedrelu6_inputs():
    list_of_inputs = []

    # Input 1
    features = np.array([-1, 0, 1, 2, 6, 7], dtype=np.int8)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([6.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "features": tf.constant(features, dtype=tf.int8),
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": "relu6_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    features = np.array([[-2, -1, 0], [1, 2, 6]], dtype=np.int8)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([6.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "features": tf.constant(features, dtype=tf.uint8),
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": "relu6_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    features = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    min_features = np.array([1.0], dtype=np.float32)
    max_features = np.array([7.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "features": tf.constant(features, dtype=tf.int32),
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": "relu6_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    features = np.array([10, 20, 30, 40], dtype=np.int16)
    min_features = np.array([5.0], dtype=np.float32)
    max_features = np.array([35.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "features": tf.constant(features, dtype=tf.int16),
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": "relu6_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    features = np.array([0, 0, 10, 20], dtype=np.uint16)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([25.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "features":  tf.constant(features, dtype=tf.uint16),
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": "relu6_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    features = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int8)
    min_features = np.array([-2.0], dtype=np.float32)
    max_features = np.array([5.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "features": tf.constant(features, dtype=tf.int8),
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": "relu6_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    features = np.array([1, 2, 3, 4, 5, 6, 6, 6], dtype=np.uint8)
    min_features = np.array([1.0], dtype=np.float32)
    max_features = np.array([6.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "features": tf.constant(features, dtype=tf.uint8),
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": "relu6_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    features = np.array([[[1, 2], [3, 4]], [[5, 6], [6, 6]]], dtype=np.int32)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([6.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "features": tf.constant(features, dtype=tf.int32),
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": "relu6_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    features = np.array([-1, 2, 3], dtype=np.int16)
    min_features = np.array([-1.0], dtype=np.float32)
    max_features = np.array([7.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "features": tf.constant(features, dtype=tf.int16),
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": "relu6_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    features = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint16)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([6.0], dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "features": tf.constant(features, dtype=tf.uint16),
        "min_features": min_features,
        "max_features": max_features,
        "out_type": out_type,
        "name": "relu6_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedRelu6"] = tf_raw_ops_quantizedrelu6_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedRelu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedRelu6'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedRelu6', generated_inputs['tf.raw_ops.QuantizedRelu6'], lib="tf", suffix=0)
