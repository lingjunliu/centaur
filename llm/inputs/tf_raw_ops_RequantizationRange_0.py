
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_requantization_range_inputs():
    list_of_inputs = []

    # Input 1: qint8
    input_tensor = np.array([-10, -5, 0, 5, 10], dtype=np.int8)
    input_min_tensor = np.array([-20.0], dtype=np.float32)
    input_max_tensor = np.array([20.0], dtype=np.float32)
    name = "qint8_range1"
    input_dict = {"input": input_tensor, "input_min": input_min_tensor, "input_max": input_max_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: quint8
    input_tensor = np.array([0, 5, 10, 15, 20], dtype=np.uint8)
    input_min_tensor = np.array([0.0], dtype=np.float32)
    input_max_tensor = np.array([25.0], dtype=np.float32)
    name = "quint8_range1"
    input_dict = {"input": input_tensor, "input_min": input_min_tensor, "input_max": input_max_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: qint32
    input_tensor = np.array([-1000, -500, 0, 500, 1000], dtype=np.int32)
    input_min_tensor = np.array([-2000.0], dtype=np.float32)
    input_max_tensor = np.array([2000.0], dtype=np.float32)
    name = "qint32_range1"
    input_dict = {"input": input_tensor, "input_min": input_min_tensor, "input_max": input_max_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: qint16
    input_tensor = np.array([-100, -50, 0, 50, 100], dtype=np.int16)
    input_min_tensor = np.array([-200.0], dtype=np.float32)
    input_max_tensor = np.array([200.0], dtype=np.float32)
    name = "qint16_range1"
    input_dict = {"input": input_tensor, "input_min": input_min_tensor, "input_max": input_max_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: quint16
    input_tensor = np.array([0, 50, 100, 150, 200], dtype=np.uint16)
    input_min_tensor = np.array([0.0], dtype=np.float32)
    input_max_tensor = np.array([250.0], dtype=np.float32)
    name = "quint16_range1"
    input_dict = {"input": input_tensor, "input_min": input_min_tensor, "input_max": input_max_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: qint8 with different range
    input_tensor = np.array([-128, 0, 127], dtype=np.int8)
    input_min_tensor = np.array([-10.0], dtype=np.float32)
    input_max_tensor = np.array([10.0], dtype=np.float32)
    name = "qint8_range2"
    input_dict = {"input": input_tensor, "input_min": input_min_tensor, "input_max": input_max_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: qint32 with large numbers
    input_tensor = np.array([-2147483648, 0, 2147483647], dtype=np.int32)
    input_min_tensor = np.array([-1e10], dtype=np.float32)
    input_max_tensor = np.array([1e10], dtype=np.float32)
    name = "qint32_range2"
    input_dict = {"input": input_tensor, "input_min": input_min_tensor, "input_max": input_max_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: qint16 with same min and max
    input_tensor = np.array([-32768, 0, 32767], dtype=np.int16)
    input_min_tensor = np.array([5.0], dtype=np.float32)
    input_max_tensor = np.array([5.0], dtype=np.float32)
    name = "qint16_range2"
    input_dict = {"input": input_tensor, "input_min": input_min_tensor, "input_max": input_max_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: quint16 with large range
    input_tensor = np.array([0, 65535], dtype=np.uint16)
    input_min_tensor = np.array([0.0], dtype=np.float32)
    input_max_tensor = np.array([1e5], dtype=np.float32)
    name = "quint16_range2"
    input_dict = {"input": input_tensor, "input_min": input_min_tensor, "input_max": input_max_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: qint8 with a name that is None
    input_tensor = np.array([-10, -5, 0, 5, 10], dtype=np.int8)
    input_min_tensor = np.array([-20.0], dtype=np.float32)
    input_max_tensor = np.array([20.0], dtype=np.float32)
    name = None
    input_dict = {"input": input_tensor, "input_min": input_min_tensor, "input_max": input_max_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
