
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_relu6_inputs():
    list_of_inputs = []

    # Input 1: Basic quint8 example, all values >=0
    features = np.array([0, 1, 2, 6, 6, 6, 7], dtype=np.uint8)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([7.0], dtype=np.float32)
    out_type = tf.quint8
    name = "relu6_1"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.quint8),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: quint8 example with different range, all values >=0
    features = np.array([0, 3, 5, 6, 7, 8, 10], dtype=np.uint8) #Making values non-negative for uint8
    min_features = np.array([0.0], dtype=np.float32) #Setting min to 0
    max_features = np.array([10.0], dtype=np.float32)
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

    # Input 3: quint16 example
    features = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint16)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([5.0], dtype=np.float32)
    out_type = tf.quint16
    name = "relu6_3"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.quint16),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: qint16 with negative values
    features = np.array([-5, -2, 0, 1, 2, 3, 5], dtype=np.int16)
    min_features = np.array([-5.0], dtype=np.float32)
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

    # Input 5: quint8 with 2D input, all values >=0
    features = np.array([[0, 1, 0], [1, 2, 6]], dtype=np.uint8) #Making values non-negative for uint8
    min_features = np.array([0.0], dtype=np.float32) #Making min value 0 for uint8
    max_features = np.array([7.0], dtype=np.float32)
    out_type = tf.quint8
    name = "relu6_5"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.quint8),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: qint32
    features = np.array([-5, -2, 0, 3, 5, 6, 10], dtype=np.int32)
    min_features = np.array([-5.0], dtype=np.float32)
    max_features = np.array([10.0], dtype=np.float32)
    out_type = tf.qint32
    name = "relu6_6"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.qint32),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: quint16 with different min max
    features = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint16)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([8.0], dtype=np.float32)
    out_type = tf.quint16
    name = "relu6_7"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.quint16),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: qint8 with min and max close to 0
    features = np.array([-1, 0, 1, 2, 3, 4], dtype=np.int8)
    min_features = np.array([-1.0], dtype=np.float32)
    max_features = np.array([4.0], dtype=np.float32)
    out_type = tf.qint8
    name = "relu6_8"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.qint8),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: quint8 all >=0
    features = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    min_features = np.array([0.0], dtype=np.float32)
    max_features = np.array([4.0], dtype=np.float32)
    out_type = tf.quint8
    name = "relu6_9"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.quint8),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: qint32 1d array
    features = np.array([-2, 0, 1, 2, 3, 4, 5, 6, 8, 9], dtype=np.int32) #added -2
    min_features = np.array([-2.0], dtype=np.float32) #min -2
    max_features = np.array([9.0], dtype=np.float32)
    out_type = tf.qint32
    name = "relu6_10"

    input_dict = {
        "features": tf.convert_to_tensor(features, dtype=tf.qint32),
        "min_features": tf.convert_to_tensor(min_features, dtype=tf.float32),
        "max_features": tf.convert_to_tensor(max_features, dtype=tf.float32),
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedRelu6"] = tf_raw_ops_quantized_relu6_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedRelu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedRelu6'.")

check_valid('tf.raw_ops.QuantizedRelu6', generated_inputs['tf.raw_ops.QuantizedRelu6'], lib="tf", suffix=0)
