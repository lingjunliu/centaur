
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    name = "xor_op1"
    input_dict = {"x": tf.constant(x.tolist()), "y": tf.constant(y.tolist()), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[5, 6], [7, 8]], dtype=np.int16)
    name = "xor_op2"
    input_dict = {"x": tf.constant(x.tolist()), "y": tf.constant(y.tolist()), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([10, 20, 30, 40], dtype=np.int64)
    y = np.array([5, 15, 25, 35], dtype=np.int64)
    name = "xor_op3"
    input_dict = {"x": tf.constant(x.tolist()), "y": tf.constant(y.tolist()), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([255, 128, 64, 0], dtype=np.uint8)
    y = np.array([1, 2, 4, 8], dtype=np.uint8)
    name = "xor_op4"
    input_dict = {"x": tf.constant(x.tolist()), "y": tf.constant(y.tolist()), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[1, 0], [0, 1]], dtype=np.uint16)
    y = np.array([[0, 1], [1, 0]], dtype=np.uint16)
    name = "xor_op5"
    input_dict = {"x": tf.constant(x.tolist()), "y": tf.constant(y.tolist()), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    y = np.array([5, 4, 3, 2, 1], dtype=np.uint32)
    name = "xor_op6"
    input_dict = {"x": tf.constant(x.tolist()), "y": tf.constant(y.tolist()), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1, 10, 100], dtype=np.int8)
    y = np.array([100, 10, 1], dtype=np.int8)
    name = "xor_op7"
    input_dict = {"x": tf.constant(x.tolist()), "y": tf.constant(y.tolist()), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([0xFFFFFFFF, 0x00000000], dtype=np.uint32)
    y = np.array([0x00000000, 0xFFFFFFFF], dtype=np.uint32)
    name = "xor_op8"
    input_dict = {"x": tf.constant(x.tolist()), "y": tf.constant(y.tolist()), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1], dtype=np.int32)
    name = None
    input_dict = {"x": tf.constant(x.tolist()), "y": tf.constant(y.tolist()), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([123456789, 987654321], dtype=np.int64)
    y = np.array([987654321, 123456789], dtype=np.int64)
    name = "xor_op10"
    input_dict = {"x": tf.constant(x.tolist()), "y": tf.constant(y.tolist()), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.bitwise.bitwise_xor"] = tf_bitwise_bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.bitwise.bitwise_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.bitwise_xor'.")

check_valid('tf.bitwise.bitwise_xor', generated_inputs['tf.bitwise.bitwise_xor'], lib="tf", suffix=0)
