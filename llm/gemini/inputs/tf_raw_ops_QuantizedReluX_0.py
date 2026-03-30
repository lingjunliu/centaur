
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_quantizedrelux_inputs():
    list_of_inputs = []

    # Input 1: qint8
    features = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.qint8)
    max_value = tf.constant(5.0, dtype=tf.float32)
    min_features = tf.constant(0.0, dtype=tf.float32)
    max_features = tf.constant(255.0, dtype=tf.float32)
    out_type = tf.qint8
    name = "relu_qint8_1"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: qint8
    features = tf.constant([[-1, 2, -3], [4, -5, 6]], dtype=tf.qint8)
    max_value = tf.constant(3.0, dtype=tf.float32)
    min_features = tf.constant(-128.0, dtype=tf.float32)
    max_features = tf.constant(127.0, dtype=tf.float32)
    out_type = tf.qint8
    name = "relu_qint8_2"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: qint32
    features = tf.constant([[100, 200, 300], [400, 500, 600]], dtype=tf.qint32)
    max_value = tf.constant(400.0, dtype=tf.float32)
    min_features = tf.constant(-2147483648.0, dtype=tf.float32)
    max_features = tf.constant(2147483647.0, dtype=tf.float32)
    out_type = tf.qint32
    name = "relu_qint32_1"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: qint16
    features = tf.constant([[-100, 200, -300], [400, -500, 600]], dtype=tf.qint16)
    max_value = tf.constant(500.0, dtype=tf.float32)
    min_features = tf.constant(-32768.0, dtype=tf.float32)
    max_features = tf.constant(32767.0, dtype=tf.float32)
    out_type = tf.qint16
    name = "relu_qint16_1"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: quint16
    features = tf.constant([[100, 200, 300], [400, 500, 600]], dtype=tf.quint16)
    max_value = tf.constant(500.0, dtype=tf.float32)
    min_features = tf.constant(0.0, dtype=tf.float32)
    max_features = tf.constant(65535.0, dtype=tf.float32)
    out_type = tf.quint16
    name = "relu_quint16_1"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D qint8
    features = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.qint8)
    max_value = tf.constant(6.0, dtype=tf.float32)
    min_features = tf.constant(0.0, dtype=tf.float32)
    max_features = tf.constant(255.0, dtype=tf.float32)
    out_type = tf.qint8
    name = "relu_qint8_3d_1"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D qint8 negative values
    features = tf.constant([[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]], dtype=tf.qint8)
    max_value = tf.constant(5.0, dtype=tf.float32)
    min_features = tf.constant(-128.0, dtype=tf.float32)
    max_features = tf.constant(127.0, dtype=tf.float32)
    out_type = tf.qint8
    name = "relu_qint8_3d_2"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different max_value qint8
    features = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.qint8)
    max_value = tf.constant(10.0, dtype=tf.float32)
    min_features = tf.constant(0.0, dtype=tf.float32)
    max_features = tf.constant(255.0, dtype=tf.float32)
    out_type = tf.qint8
    name = "relu_qint8_1_maxvalue"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero maxValue qint8
    features = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.qint8)
    max_value = tf.constant(0.0, dtype=tf.float32)
    min_features = tf.constant(0.0, dtype=tf.float32)
    max_features = tf.constant(255.0, dtype=tf.float32)
    out_type = tf.qint8
    name = "relu_qint8_zero_maxvalue"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D qint8
    features = tf.constant([1, 2, 3, 4, 5], dtype=tf.qint8)
    max_value = tf.constant(3.0, dtype=tf.float32)
    min_features = tf.constant(0.0, dtype=tf.float32)
    max_features = tf.constant(255.0, dtype=tf.float32)
    out_type = tf.qint8
    name = "relu_qint8_1d"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: quint8
    features = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.quint8)
    max_value = tf.constant(5.0, dtype=tf.float32)
    min_features = tf.constant(0.0, dtype=tf.float32)
    max_features = tf.constant(255.0, dtype=tf.float32)
    out_type = tf.quint8
    name = "relu_quint8_1"
    input_dict = {"features": features, "max_value": max_value, "min_features": min_features, "max_features": max_features, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedReluX"] = tf_raw_ops_quantizedrelux_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedReluX' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedReluX'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedReluX', generated_inputs['tf.raw_ops.QuantizedReluX'], lib="tf", suffix=0)
