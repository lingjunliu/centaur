
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_ceil_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0], dtype=np.float32)
    name = "ceil_example_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[-1.0, -0.5], [0.0, 1.5]], dtype=np.float64)
    name = "ceil_example_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    name = "ceil_example_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[-2.5, -1.5], [0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    name = "ceil_example_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    x = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    name = "ceil_example_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=np.float32)
    name = "ceil_example_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    name = "ceil_example_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[-1.0, -0.5], [0.0, 1.5]], dtype=np.float32)
    name = "ceil_example_11"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0], dtype=np.float64)
    name = "ceil_example_12"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    name = "ceil_example_13"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.ceil"] = tf_math_ceil_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.ceil' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.ceil'.")

check_valid('tf.math.ceil', generated_inputs['tf.math.ceil'], lib="tf", suffix=0)
