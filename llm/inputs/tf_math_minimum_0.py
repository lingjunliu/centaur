
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_minimum_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.5, 2.5, 2.8], dtype=np.float32)
    name = "minimum_example_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1, 0, 1], dtype=np.int32)
    y = np.array([0, -1, 2], dtype=np.int32)
    name = "minimum_example_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[0, 3], [2, 5]], dtype=np.float64)
    name = "minimum_example_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([1, 2, 3, 4], dtype=np.int16)
    y = np.array([5], dtype=np.int16)
    name = "minimum_example_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    y = np.array([[0.5, -1.0], [-2.0, 3.0]], dtype=np.float32)
    name = "minimum_example_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([10, 20, 30], dtype=np.uint8)
    y = np.array([5, 25, 35], dtype=np.uint8)
    name = "minimum_example_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([-1, -2, -3], dtype=np.int64)
    name = "minimum_example_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    y = np.array([0.2], dtype=np.float32)
    name = "minimum_example_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([100, 200], dtype=np.int32)
    y = np.array([50, 250], dtype=np.int32)
    name = "minimum_example_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    y = np.array([[[0, 3], [2, 5]], [[4, 7], [6, 9]]], dtype=np.int32)
    name = "minimum_example_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.minimum"] = tf_math_minimum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.minimum'.")

check_valid('tf.math.minimum', generated_inputs['tf.math.minimum'], lib="tf", suffix=0)
