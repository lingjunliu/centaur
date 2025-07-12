
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_is_inf_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([5.0, np.inf, 6.8, -np.inf], dtype=np.float32)
    name = "input_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[np.inf, 2.0], [3.0, -np.inf]], dtype=np.float64)
    name = "input_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    name = "input_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    name = "input_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([np.inf], dtype=np.float32)
    name = "input_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    x = np.array([-np.inf], dtype=np.float64)
    name = "input_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[-np.inf, 1.0], [2.0, np.inf]], dtype=np.float32)
    name = "input_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1.0, -1.0, np.inf, -np.inf], dtype=np.float64)
    name = "input_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[[np.inf, 1.0], [2.0, -np.inf]], [[3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    name = "input_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([], dtype=np.float64)
    name = "input_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.is_inf"] = tf_math_is_inf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.is_inf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.is_inf'.")

check_valid('tf.math.is_inf', generated_inputs['tf.math.is_inf'], lib="tf", suffix=0)
