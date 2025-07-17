
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_requantization_range_inputs():
    list_of_inputs = []

    # Input 2: quint8
    input2 = np.array([0, 5, 10, 15, 20], dtype=np.uint8)
    input_min2 = np.array([0.0], dtype=np.float32)
    input_max2 = np.array([20.0], dtype=np.float32)
    input_dict2 = {"input":  tf.constant(input2, dtype=tf.quint8), "input_min": tf.constant(input_min2), "input_max": tf.constant(input_max2), "name": "quint8_example"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: qint32
    input3 = np.array([-1000, -500, 0, 500, 1000], dtype=np.int32)
    input_min3 = np.array([-1000.0], dtype=np.float32)
    input_max3 = np.array([1000.0], dtype=np.float32)
    input_dict3 = {"input": tf.constant(input3, dtype=tf.qint32), "input_min": tf.constant(input_min3), "input_max": tf.constant(input_max3), "name": "qint32_example"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: qint16
    input4 = np.array([-100, -50, 0, 50, 100], dtype=np.int16)
    input_min4 = np.array([-100.0], dtype=np.float32)
    input_max4 = np.array([100.0], dtype=np.float32)
    input_dict4 = {"input": tf.constant(input4, dtype=tf.qint16), "input_min": tf.constant(input_min4), "input_max": tf.constant(input_max4), "name": "qint16_example"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: quint16
    input5 = np.array([0, 50, 100, 150, 200], dtype=np.uint16)
    input_min5 = np.array([0.0], dtype=np.float32)
    input_max5 = np.array([200.0], dtype=np.float32)
    input_dict5 = {"input": tf.constant(input5, dtype=tf.quint16), "input_min": tf.constant(input_min5), "input_max": tf.constant(input_max5), "name": "quint16_example"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 7: Different min/max for quint8
    input7 = np.array([50, 75, 100, 125, 150], dtype=np.uint8)
    input_min7 = np.array([50.0], dtype=np.float32)
    input_max7 = np.array([150.0], dtype=np.float32)
    input_dict7 = {"input": tf.constant(input7, dtype=tf.quint8), "input_min": tf.constant(input_min7), "input_max": tf.constant(input_max7), "name": "quint8_diff_example"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 9: Negative range for qint32
    input9 = np.array([-5000, -2500, 0, 1000, 2000], dtype=np.int32)
    input_min9 = np.array([-5000.0], dtype=np.float32)
    input_max9 = np.array([2000.0], dtype=np.float32)
    input_dict9 = {"input": tf.constant(input9, dtype=tf.qint32), "input_min": tf.constant(input_min9), "input_max": tf.constant(input_max9), "name": "qint32_neg_example"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

   # Input 10: Small range for quint16
    input10 = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    input_min10 = np.array([1.0], dtype=np.float32)
    input_max10 = np.array([5.0], dtype=np.float32)
    input_dict10 = {"input": tf.constant(input10, dtype=tf.quint16), "input_min": tf.constant(input_min10), "input_max": tf.constant(input_max10), "name": "quint16_small_example"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Reshape input tensors for variety
    input11 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_min11 = np.array([-10.0], dtype=np.float32)
    input_max11 = np.array([10.0], dtype=np.float32)
    input_dict11 = {"input": tf.constant(input11, dtype=tf.qint32), "input_min": tf.constant(input_min11), "input_max": tf.constant(input_max11), "name": "qint32_reshaped"}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12: scalar min/max
    input12 = np.array([1, 2, 3], dtype=np.uint16)
    input_min12 = np.array(1.0, dtype=np.float32)
    input_max12 = np.array(5.0, dtype=np.float32)
    input_dict12 = {"input": tf.constant(input12, dtype=tf.quint16), "input_min": tf.constant(input_min12), "input_max": tf.constant(input_max12), "name": "quint16_scalar_minmax"}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    # Input 13: different shape for min/max. must be broadcastable
    input13 = np.array([1, 2, 3, 4], dtype=np.int32)
    input_min13 = np.array([1.0], dtype=np.float32)
    input_max13 = np.array([5.0], dtype=np.float32)
    input_dict13 = {"input": tf.constant(input13, dtype=tf.qint32), "input_min": tf.constant(input_min13), "input_max": tf.constant(input_max13), "name": "qint32_diff_minmax_shape"}
    list_of_inputs.append(copy.deepcopy(input_dict13))

    # Input 14: Empty tensor
    input14 = np.array([], dtype=np.int32)
    input_min14 = np.array([1.0], dtype=np.float32)
    input_max14 = np.array([5.0], dtype=np.float32)
    input_dict14 = {"input": tf.constant(input14, dtype=tf.qint32), "input_min": tf.constant(input_min14), "input_max": tf.constant(input_max14), "name": "qint32_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict14))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RequantizationRange"] = tf_raw_ops_requantization_range_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RequantizationRange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RequantizationRange'.")

check_valid('tf.raw_ops.RequantizationRange', generated_inputs['tf.raw_ops.RequantizationRange'], lib="tf", suffix=0)
