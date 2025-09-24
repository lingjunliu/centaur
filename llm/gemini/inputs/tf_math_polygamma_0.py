
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_polygamma_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array(0.0, dtype=np.float32)
    x = np.array(1.0, dtype=np.float32)
    name = "polygamma_1"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array(1.0, dtype=np.float64)
    x = np.array(2.5, dtype=np.float64)
    name = "polygamma_2"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array(2.0, dtype=np.float32)
    x = np.arange(1.0, 5.0, dtype=np.float32)
    name = "polygamma_3"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[0, 1], [2, 3]], dtype=np.float32)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    name = "polygamma_4"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([0, 1, 2], dtype=np.float64)
    x = np.array([3.0, 4.0, 5.0], dtype=np.float64)
    name = "polygamma_5"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array(3, dtype=np.float32)
    x = np.array(5, dtype=np.float32)
    name = "polygamma_6"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Multi-dimensional a and x
    a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    name = "polygamma_7"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different values for a and x
    a = np.array([0, 1, 2], dtype=np.float64)
    x = np.array(5.0, dtype=np.float64)
    name = "polygamma_8"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Higher values
    a = np.array(10.0, dtype=np.float32)
    x = np.array(20.0, dtype=np.float32)
    name = "polygamma_9"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: scalar tensors
    a = np.array(5.0, dtype=np.float64)
    x = np.array(1.0, dtype=np.float64)
    name = "polygamma_10"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.polygamma"] = tf_math_polygamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.polygamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.polygamma'.")

check_valid('tf.math.polygamma', generated_inputs['tf.math.polygamma'], lib="tf", suffix=0)
