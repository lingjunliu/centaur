
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_digamma_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "digamma_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    name = "digamma_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, scalar
    x = np.array(0.75, dtype=np.float16)
    name = "digamma_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float16, 3D array
    x = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float16)
    name = "digamma_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, mixed positive and negative
    x = np.array([-1.5, 0.5, 2.5, -3.5], dtype=np.float32)
    name = "digamma_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, large values
    x = np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)
    name = "digamma_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, 2D with different values
    x = np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float16)
    name = "digamma_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16, scalar
    x = np.array(2.0, dtype=np.float16)
    name = "digamma_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, 4D array
    x = np.random.rand(2, 2, 2, 2).astype(np.float32)
    name = "digamma_11"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, 1D array
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    name = "digamma_12"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.digamma"] = tf_math_digamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.digamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.digamma'.")

check_valid('tf.math.digamma', generated_inputs['tf.math.digamma'], lib="tf", suffix=0)
