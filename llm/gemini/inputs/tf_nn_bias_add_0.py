
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_bias_add_inputs():
    list_of_inputs = []

    # Input 1: Basic case with N...C format (default)
    value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    data_format = None
    name = "bias_add_1"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NC... format
    value = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    bias = np.array([0.1, 0.2], dtype=np.float32)
    data_format = "NC..."
    name = "bias_add_2"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data type (int32)
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    bias = np.array([1, 2, 3], dtype=np.int32)
    data_format = None
    name = "bias_add_3"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    value = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    bias = np.array([-0.1, 0.2, -0.3], dtype=np.float32)
    data_format = None
    name = "bias_add_4"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D tensor with N...C format
    value = np.random.rand(2, 3, 4, 5).astype(np.float32)
    bias = np.random.rand(5).astype(np.float32)
    data_format = None
    name = "bias_add_5"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Removing NC... format with incorrect bias dimension
    # value = np.random.rand(2, 5, 3, 4).astype(np.float32)
    # bias = np.random.rand(2).astype(np.float32)
    # data_format = "NC..."
    # name = "bias_add_6"
    # input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    bias = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    data_format = None
    name = "bias_add_7"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    bias = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    data_format = None
    name = "bias_add_8"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64 type
    value = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=np.complex64)
    bias = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=np.complex64)
    data_format = None
    name = "bias_add_9"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    bias = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    data_format = None
    name = "bias_add_10"
    input_dict = {"value": value, "bias": bias, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.bias_add"] = tf_nn_bias_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.bias_add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.bias_add'.")

check_valid('tf.nn.bias_add', generated_inputs['tf.nn.bias_add'], lib="tf", suffix=0)
