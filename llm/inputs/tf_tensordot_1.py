
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_tensordot_inputs():
    list_of_inputs = []

    # Input 1
    a = np.arange(24).reshape((2, 3, 4)).astype(np.float32)
    b = np.arange(12).reshape((3, 4)).astype(np.float32)
    axes = [[1, 2], [0, 1]]
    name = "contraction1"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.arange(6).reshape((1, 2, 3)).astype(np.float32)
    b = np.arange(15).reshape((3, 5)).astype(np.float32)
    axes = [[2], [0]]
    name = "contraction2"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.arange(12).reshape((3, 4)).astype(np.float32)
    b = np.arange(20).reshape((4, 5)).astype(np.float32)
    axes = [[1], [0]]
    name = "contraction3"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.arange(12).reshape((3, 4)).astype(np.float32)
    b = np.arange(12).reshape((4, 3)).astype(np.float32)
    axes = [[1], [0]]
    name = "contraction4"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.arange(8).reshape((2, 2, 2)).astype(np.float32)
    b = np.arange(16).reshape((2, 4, 2)).astype(np.float32)
    axes = [[0, 2], [0, 2]]
    name = "contraction5"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    a = np.arange(10).reshape((2, 5)).astype(np.float32)
    b = np.arange(15).reshape((5, 3)).astype(np.float32)
    axes = [[1], [0]]
    name = "contraction6"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.arange(24).reshape((2, 3, 4)).astype(np.float32)
    b = np.arange(12).reshape((4, 3)).astype(np.float32)
    axes = [[2, 1], [0, 1]]
    name = "contraction7"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.arange(9).reshape((3, 3)).astype(np.float32)
    b = np.arange(9).reshape((3, 3)).astype(np.float32)
    axes = [[0], [0]]
    name = "contraction8"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.arange(8).reshape((2, 2, 2)).astype(np.float32)
    b = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    axes = [[1,2],[1,2]]
    name = "contraction9"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    a = np.arange(16).reshape((2, 2, 2, 2)).astype(np.float32)
    b = np.arange(24).reshape((2, 3, 4)).astype(np.float32)
    axes = [[0], [0]]
    name = "contraction10"
    input_dict = {"a": a, "b": b, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.tensordot_1"] = tf_tensordot_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.tensordot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tensordot_1'.")

check_valid('tf.tensordot', generated_inputs['tf.tensordot_1'], lib="tf", suffix=1)
