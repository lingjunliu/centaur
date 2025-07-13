
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_lgamma_inputs():
    list_of_inputs = []

    # Input 1: Simple float32
    x = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    name = "lgamma_example_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple float64
    x = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64)
    name = "lgamma_example_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values float32
    x = np.array([-0.5, -1.0, -2.0, -3.0], dtype=np.float32)
    name = "lgamma_example_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values float64
    x = np.array([-0.5, -1.0, -2.0, -3.0], dtype=np.float64)
    name = "lgamma_example_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive values float32
    x = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float32)
    name = "lgamma_example_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array float32
    x = np.array([[0.5, 1.0], [2.0, 3.0]], dtype=np.float32)
    name = "lgamma_example_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array float64
    x = np.array([[0.5, 1.0], [2.0, 3.0]], dtype=np.float64)
    name = "lgamma_example_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small positive values
    x = np.array([0.001, 0.01, 0.1], dtype=np.float32)
    name = "lgamma_example_11"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Values close to zero (negative)
    x = np.array([-0.001, -0.01, -0.1], dtype=np.float64)
    name = "lgamma_example_12"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half
    x = np.array([0.5, 1.0, 2.0], dtype=np.float16)
    name = "lgamma_example_13"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Lgamma"] = tf_raw_ops_lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Lgamma'.")

check_valid('tf.raw_ops.Lgamma', generated_inputs['tf.raw_ops.Lgamma'], lib="tf", suffix=0)
