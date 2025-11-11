
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_bitwise_xor_inputs():
    list_of_inputs = []

    x = np.array([0, 5, 3, 14], dtype=np.int8)
    y = np.array([5, 0, 7, 11], dtype=np.int8)
    name = "xor_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.int16)
    y = np.array([5, 15, 25], dtype=np.int16)
    name = "xor_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    name = "xor_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([100, 200, 300], dtype=np.int64)
    y = np.array([50, 150, 250], dtype=np.int64)
    name = "xor_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3, 4], dtype=np.uint8)
    y = np.array([5, 6, 7, 8], dtype=np.uint8)
    name = "xor_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.uint32)
    y = np.array([5, 15, 25], dtype=np.uint32)
    name = "xor_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.bitwise.bitwise_xor"] = tf_bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.bitwise.bitwise_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.bitwise_xor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.bitwise.bitwise_xor', generated_inputs['tf.bitwise.bitwise_xor'], lib="tf", suffix=0)
