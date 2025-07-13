
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_real_div_inputs():
    list_of_inputs = []

    # Input 1: Basic float division
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.5, 2.0, 1.5], dtype=np.float32)
    name = "real_div_1"
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer division
    x = np.array([4, 8, 12], dtype=np.int32)
    y = np.array([2, 4, 3], dtype=np.int32)
    name = "real_div_2"
    x = tf.convert_to_tensor(x, dtype=tf.int32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    name = "real_div_3"
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    name = "real_div_4"
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex number division
    x = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    y = np.array([0.5 + 0.5j, 1 + 1j], dtype=np.complex64)
    name = "real_div_5"
    x = tf.convert_to_tensor(x, dtype=tf.complex64)
    y = tf.convert_to_tensor(y, dtype=tf.complex64)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dtype (bfloat16)
    x = np.array([1.0, 2.0], dtype=np.float16)
    y = np.array([0.5, 1.0], dtype=np.float16)
    name = "real_div_6"
    x = tf.convert_to_tensor(x, dtype=tf.float16)
    y = tf.convert_to_tensor(y, dtype=tf.float16)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero division (avoiding actual zero to prevent errors)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.001, 0.002, 0.003], dtype=np.float32)
    name = "real_div_7"
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large numbers
    x = np.array([1e9, 2e9], dtype=np.float32)
    y = np.array([1e6, 2e6], dtype=np.float32)
    name = "real_div_8"
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative numbers
    x = np.array([-1.0, -2.0], dtype=np.float32)
    y = np.array([0.5, -1.0], dtype=np.float32)
    name = "real_div_9"
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8
    x = np.array([255, 128, 64], dtype=np.uint8)
    y = np.array([5, 4, 2], dtype=np.uint8)
    name = "real_div_10"
    x = tf.convert_to_tensor(x, dtype=tf.uint8)
    y = tf.convert_to_tensor(y, dtype=tf.uint8)
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RealDiv"] = tf_raw_ops_real_div_inputs()
import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RealDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RealDiv'.")

check_valid('tf.raw_ops.RealDiv', generated_inputs['tf.raw_ops.RealDiv'], lib="tf", suffix=0)
