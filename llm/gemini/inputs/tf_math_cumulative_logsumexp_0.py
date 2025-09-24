
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_cumulative_logsumexp_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = 0
    exclusive = False
    reverse = False
    name = None
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 0
    exclusive = True
    reverse = False
    name = "test2"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 1
    exclusive = False
    reverse = True
    name = None
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 0
    exclusive = True
    reverse = True
    name = "test4"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = -1
    exclusive = False
    reverse = False
    name = None
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    axis = 0
    exclusive = True
    reverse = False
    name = "test6"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    axis = 0
    exclusive = False
    reverse = False
    name = None
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    axis = 1
    exclusive = True
    reverse = True
    name = "test8"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    exclusive = False
    reverse = True
    name = None
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 1
    exclusive = True
    reverse = False
    name = "test10"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.cumulative_logsumexp"] = tf_math_cumulative_logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.cumulative_logsumexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.cumulative_logsumexp'.")

check_valid('tf.math.cumulative_logsumexp', generated_inputs['tf.math.cumulative_logsumexp'], lib="tf", suffix=0)
