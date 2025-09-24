
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_log1p_inputs():
    list_of_inputs = []

    # Input 1
    x = tf.constant(np.array([0, 0.5, 1, 5], dtype=np.float32))
    name = "log1p_example_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = tf.constant(np.array([-0.5, 0, 1.5], dtype=np.float64))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = tf.constant(np.array([[-0.9, 0.1], [0.5, 1.0]], dtype=np.float32))
    name = "log1p_example_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = tf.constant(np.array([1e-8, 1e-7, 1e-6], dtype=np.float64))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = tf.constant(np.array([[-1e-8, -1e-7], [-1e-6, 0]], dtype=np.float32))
    name = "log1p_example_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = tf.constant(np.array([0.0], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = tf.constant(np.array([1.0], dtype=np.float64))
    name = "log1p_example_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = tf.constant(np.array([0.9999], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = tf.constant(np.array([-0.9999], dtype=np.float64))
    name = "log1p_example_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    x = tf.constant(np.array([1,2,3,4,5,6,7,8]).reshape((2,2,2)), dtype=np.float32)
    name = "log1p_example_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.log1p"] = tf_math_log1p_inputs()

for input_dict in generated_inputs["tf.math.log1p"]:
    input_dict["x"] = input_dict["x"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.log1p'.")

check_valid('tf.math.log1p', generated_inputs['tf.math.log1p'], lib="tf", suffix=0)
