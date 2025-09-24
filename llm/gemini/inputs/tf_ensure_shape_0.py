
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ensure_shape_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shape = [2, 3]
    name = "input_1"
    input_dict = {"x": x, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    shape = [5]
    name = "input_2"
    input_dict = {"x": x, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    shape = [2, 2, 2]
    name = "input_3"
    input_dict = {"x": x, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    shape = [3, 2]
    name = "input_4"
    input_dict = {"x": x, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1, 2, 3], dtype=np.int32)
    shape = [3]
    name = "input_5"
    input_dict = {"x": x, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shape = [2, 3]
    name = "input_6"
    input_dict = {"x": x, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    shape = [2, 2, 2]
    name = "input_7"
    input_dict = {"x": x, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    shape = [3, 2]
    name = "input_8"
    input_dict = {"x": x, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    shape = [2, 2, 2]
    name = "input_9"
    input_dict = {"x": x, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    shape = [4]
    name = "input_10"
    input_dict = {"x": x, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ensure_shape"] = tf_ensure_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ensure_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ensure_shape'.")

check_valid('tf.ensure_shape', generated_inputs['tf.ensure_shape'], lib="tf", suffix=0)
