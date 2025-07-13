
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_unique_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 1, 2, 4, 4, 4, 7, 8, 8])
    out_idx = tf.int32
    name = "unique_values"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([4, 5, 1, 2, 3, 3, 4, 5])
    out_idx = tf.int64
    name = None
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1, 2, 3, 4, 5])
    out_idx = tf.int32
    name = "ascending"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([5, 4, 3, 2, 1])
    out_idx = tf.int64
    name = "descending"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1, 1, 1, 1, 1])
    out_idx = tf.int32
    name = "all_same"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([])
    out_idx = tf.int64
    name = "empty"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
    out_idx = tf.int32
    name = "float_values"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([-1, -2, -1, -3, -2])
    out_idx = tf.int64
    name = "negative_values"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([0, 0, 0, 0, 0])
    out_idx = tf.int32
    name = "zeros"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1, 2, 3, 1, 2, 3])
    out_idx = tf.int64
    name = "mixed"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    x = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    out_idx = tf.int32
    name = "long_array"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.unique"] = tf_unique_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.unique'.")

check_valid('tf.unique', generated_inputs['tf.unique'], lib="tf", suffix=0)
