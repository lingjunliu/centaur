
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_erfc_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    name = "erfc_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1.0, 0.5, 1.5], dtype=np.float64)
    name = "erfc_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    name = "erfc_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[-1.0, -0.5], [1.0, 1.5]], dtype=np.float64)
    name = "erfc_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[[0.0, 0.5], [1.0, 1.5]], [[2.0, 2.5], [3.0, 3.5]]], dtype=np.float32)
    name = "erfc_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[[0.0, 0.5], [1.0, 1.5]], [[-2.0, -2.5], [3.0, 3.5]]], dtype=np.float64)
    name = "erfc_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([0.0], dtype=np.float16)
    name = "erfc_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1.0], dtype=np.float16)
    name = "erfc_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([-1.0], dtype=np.float16)
    name = "erfc_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([0.5], dtype=np.float16)
    name = "erfc_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.erfc"] = tf_math_erfc_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.erfc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.erfc'.")

check_valid('tf.math.erfc', generated_inputs['tf.math.erfc'], lib="tf", suffix=0)
