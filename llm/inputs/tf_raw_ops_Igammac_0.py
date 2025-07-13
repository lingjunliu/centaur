
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_igammac_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    name = "igammac_1"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    name = "igammac_2"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    x = np.array([0.05, 0.1, 0.15], dtype=np.float16)
    name = "igammac_3"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    x = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    name = "igammac_4"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([10.0], dtype=np.float64)
    x = np.array([5.0], dtype=np.float64)
    name = "igammac_5"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([0.5], dtype=np.float32)
    x = np.array([1.0], dtype=np.float32)
    name = "igammac_6"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    x = np.array([0.75, 1.25, 1.75], dtype=np.float32)
    name = "igammac_7"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    x = np.array([[2.5, 3.0], [3.5, 4.0]], dtype=np.float32)
    name = "igammac_8"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    x = np.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]], dtype=np.float32)
    name = "igammac_9"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([0.01, 0.02], dtype=np.float64)
    x = np.array([0.005, 0.01], dtype=np.float64)
    name = "igammac_10"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Igammac"] = tf_raw_ops_igammac_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Igammac' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Igammac'.")

check_valid('tf.raw_ops.Igammac', generated_inputs['tf.raw_ops.Igammac'], lib="tf", suffix=0)
