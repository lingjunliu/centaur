
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_zeta_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    x = np.array([2.0], dtype=np.float32)
    q = np.array([1.0], dtype=np.float32)
    input_dict = {"x": x, "q": q, "name": "zeta_example_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with float64
    x = np.array([2.0], dtype=np.float64)
    q = np.array([1.0], dtype=np.float64)
    input_dict = {"x": x, "q": q, "name": "zeta_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Array inputs with float32
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    q = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    input_dict = {"x": x, "q": q, "name": "zeta_example_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Array inputs with float64
    x = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    q = np.array([1.0, 1.5, 2.0], dtype=np.float64)
    input_dict = {"x": x, "q": q, "name": "zeta_example_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different values for x and q (float32)
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    q = np.array([0.2, 1.2, 2.2], dtype=np.float32)
    input_dict = {"x": x, "q": q, "name": "zeta_example_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different values for x and q (float64)
    x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    q = np.array([0.2, 1.2, 2.2], dtype=np.float64)
    input_dict = {"x": x, "q": q, "name": "zeta_example_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional arrays (float32)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    q = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    input_dict = {"x": x, "q": q, "name": "zeta_example_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional arrays (float64)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    q = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    input_dict = {"x": x, "q": q, "name": "zeta_example_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x values between 0 and 1 (float32)
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    q = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    input_dict = {"x": x, "q": q, "name": "zeta_example_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x values between 0 and 1 (float64)
    x = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    q = np.array([1.0, 1.5, 2.0], dtype=np.float64)
    input_dict = {"x": x, "q": q, "name": "zeta_example_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Zeta"] = tf_raw_ops_zeta_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Zeta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Zeta'.")

check_valid('tf.raw_ops.Zeta', generated_inputs['tf.raw_ops.Zeta'], lib="tf", suffix=0)
