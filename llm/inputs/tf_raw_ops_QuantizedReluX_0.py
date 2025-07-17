
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedReluX_inputs():
    list_of_inputs = []

    # Input 1
    features = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    max_value = np.array(6.0, dtype=np.float32)
    min_features = np.array(0.0, dtype=np.float32)
    max_features = np.array(10.0, dtype=np.float32)
    out_type = tf.qint8
    name = "relu_example_1"
    features = tf.constant(features, dtype=tf.qint8)
    max_value = tf.constant(max_value, dtype=tf.float32)
    min_features = tf.constant(min_features, dtype=tf.float32)
    max_features = tf.constant(max_features, dtype=tf.float32)
    input_dict = {"out_type": out_type, "name": name, "features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    features = np.array([-1, 0, 1, 2, 3], dtype=np.int8)
    max_value = np.array(4.0, dtype=np.float32)
    min_features = np.array(-2.0, dtype=np.float32)
    max_features = np.array(5.0, dtype=np.float32)
    out_type = tf.quint8
    name = "relu_example_2"
    features = tf.constant(features, dtype=tf.qint8)
    max_value = tf.constant(max_value, dtype=tf.float32)
    min_features = tf.constant(min_features, dtype=tf.float32)
    max_features = tf.constant(max_features, dtype=tf.float32)
    input_dict = {"out_type": out_type, "name": name, "features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    features = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    max_value = np.array(5.0, dtype=np.float32)
    min_features = np.array(0.0, dtype=np.float32)
    max_features = np.array(6.0, dtype=np.float32)
    out_type = tf.quint8
    name = "relu_example_3"
    features = tf.constant(features, dtype=tf.quint8)
    max_value = tf.constant(max_value, dtype=tf.float32)
    min_features = tf.constant(min_features, dtype=tf.float32)
    max_features = tf.constant(max_features, dtype=tf.float32)
    input_dict = {"out_type": out_type, "name": name, "features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    features = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    max_value = np.array(7.0, dtype=np.float32)
    min_features = np.array(-1.0, dtype=np.float32)
    max_features = np.array(8.0, dtype=np.float32)
    out_type = tf.qint32
    name = "relu_example_4"
    features = tf.constant(features, dtype=tf.qint32)
    max_value = tf.constant(max_value, dtype=tf.float32)
    min_features = tf.constant(min_features, dtype=tf.float32)
    max_features = tf.constant(max_features, dtype=tf.float32)
    input_dict = {"out_type": out_type, "name": name, "features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    features = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    max_value = np.array(8.0, dtype=np.float32)
    min_features = np.array(-2.0, dtype=np.float32)
    max_features = np.array(9.0, dtype=np.float32)
    out_type = tf.qint16
    name = "relu_example_5"
    features = tf.constant(features, dtype=tf.qint16)
    max_value = tf.constant(max_value, dtype=tf.float32)
    min_features = tf.constant(min_features, dtype=tf.float32)
    max_features = tf.constant(max_features, dtype=tf.float32)
    input_dict = {"out_type": out_type, "name": name, "features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    features = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    max_value = np.array(9.0, dtype=np.float32)
    min_features = np.array(0.0, dtype=np.float32)
    max_features = np.array(10.0, dtype=np.float32)
    out_type = tf.quint16
    name = "relu_example_6"
    features = tf.constant(features, dtype=tf.quint16)
    max_value = tf.constant(max_value, dtype=tf.float32)
    min_features = tf.constant(min_features, dtype=tf.float32)
    max_features = tf.constant(max_features, dtype=tf.float32)
    input_dict = {"out_type": out_type, "name": name, "features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    features = np.array([[-1, 2], [-3, 4]], dtype=np.int8)
    max_value = np.array(1.0, dtype=np.float32)
    min_features = np.array(-4.0, dtype=np.float32)
    max_features = np.array(4.0, dtype=np.float32)
    out_type = tf.qint8
    name = "relu_example_7"
    features = tf.constant(features, dtype=tf.qint8)
    max_value = tf.constant(max_value, dtype=tf.float32)
    min_features = tf.constant(min_features, dtype=tf.float32)
    max_features = tf.constant(max_features, dtype=tf.float32)
    input_dict = {"out_type": out_type, "name": name, "features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    features = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    max_value = np.array(7.0, dtype=np.float32)
    min_features = np.array(0.0, dtype=np.float32)
    max_features = np.array(9.0, dtype=np.float32)
    out_type = tf.quint8
    name = "relu_example_8"
    features = tf.constant(features, dtype=tf.quint8)
    max_value = tf.constant(max_value, dtype=tf.float32)
    min_features = tf.constant(min_features, dtype=tf.float32)
    max_features = tf.constant(max_features, dtype=tf.float32)
    input_dict = {"out_type": out_type, "name": name, "features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    features = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    max_value = np.array(10.0, dtype=np.float32)
    min_features = np.array(1.0, dtype=np.float32)
    max_features = np.array(11.0, dtype=np.float32)
    out_type = tf.qint32
    name = "relu_example_9"
    features = tf.constant(features, dtype=tf.quint8)
    max_value = tf.constant(max_value, dtype=tf.float32)
    min_features = tf.constant(min_features, dtype=tf.float32)
    max_features = tf.constant(max_features, dtype=tf.float32)
    input_dict = {"out_type": out_type, "name": name, "features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    features = np.array([[-1, -2], [-3, -4]], dtype=np.int16)
    max_value = np.array(0.0, dtype=np.float32)
    min_features = np.array(-5.0, dtype=np.float32)
    max_features = np.array(5.0, dtype=np.float32)
    out_type = tf.qint16
    name = "relu_example_10"
    features = tf.constant(features, dtype=tf.qint16)
    max_value = tf.constant(max_value, dtype=tf.float32)
    min_features = tf.constant(min_features, dtype=tf.float32)
    max_features = tf.constant(max_features, dtype=tf.float32)
    input_dict = {"out_type": out_type, "name": name, "features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedReluX"] = tf_raw_ops_QuantizedReluX_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedReluX' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedReluX'.")

check_valid('tf.raw_ops.QuantizedReluX', generated_inputs['tf.raw_ops.QuantizedReluX'], lib="tf", suffix=0)
