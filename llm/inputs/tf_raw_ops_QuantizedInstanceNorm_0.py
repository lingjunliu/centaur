
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedInstanceNorm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with quint8
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.uint8)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(255.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With output_range_given True
    x = np.array([[[[10, 20], [30, 40]]]]).astype(np.uint8)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(100.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": True,
        "given_y_min": 1.0,
        "given_y_max": 5.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different variance_epsilon
    x = np.array([[[[1, 2], [3, 4]]]]).astype(np.uint8)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(255.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-03,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different min_separation
    x = np.array([[[[1, 2], [3, 4]]]]).astype(np.uint8)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(255.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.01,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger input
    x = np.random.randint(0, 256, size=(2, 2, 4, 4)).astype(np.uint8)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(255.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: qint8 input
    x = np.array([[[[1, -2], [-3, 4]]]]).astype(np.int8)
    x_min = np.array(-128.0, dtype=np.float32)
    x_max = np.array(127.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: qint32 input
    x = np.array([[[[1, -2], [-3, 4]]]]).astype(np.int32)
    x_min = np.array(-2147483648.0, dtype=np.float32)
    x_max = np.array(2147483647.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: qint16 input
    x = np.array([[[[1, -2], [-3, 4]]]]).astype(np.int16)
    x_min = np.array(-32768.0, dtype=np.float32)
    x_max = np.array(32767.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: quint16 input
    x = np.array([[[[1, 2], [3, 4]]]]).astype(np.uint16)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(65535.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Name specified
    x = np.array([[[[1, 2], [3, 4]]]]).astype(np.uint8)
    x_min = np.array(0.0, dtype=np.float32)
    x_max = np.array(255.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "x_min": x_min,
        "x_max": x_max,
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": "instance_norm_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedInstanceNorm"] = tf_raw_ops_QuantizedInstanceNorm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedInstanceNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedInstanceNorm'.")

check_valid('tf.raw_ops.QuantizedInstanceNorm', generated_inputs['tf.raw_ops.QuantizedInstanceNorm'], lib="tf", suffix=0)
