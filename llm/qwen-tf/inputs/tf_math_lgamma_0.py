
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def generate_lgamma_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([0, 0.5, 1, 4.5, -4, -5.6], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test1"
    }
    list_of_inputs.append(input_dict)

    # Input 2, valid
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test2"
    }
    list_of_inputs.append(input_dict)

    # Input 3, valid
    x = np.array([0.1, 0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test3"
    }
    list_of_inputs.append(input_dict)

    # Input 4, valid
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test4"
    }
    list_of_inputs.append(input_dict)

    # Input 5, valid
    x = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test5"
    }
    list_of_inputs.append(input_dict)

    # Input 6, valid
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test6"
    }
    list_of_inputs.append(input_dict)

    # Input 7, valid
    x = np.array([-0.5, -1.5, -2.5], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test7"
    }
    list_of_inputs.append(input_dict)

    # Input 8, valid
    x = np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test8"
    }
    list_of_inputs.append(input_dict)

    # Input 9, valid
    x = np.array([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test9"
    }
    list_of_inputs.append(input_dict)

    # Input 10, valid
    x = np.array([5.5, 6.5, 7.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test10"
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.math.lgamma"] = generate_lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.lgamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.lgamma', generated_inputs['tf.math.lgamma'], lib="tf", suffix=0)
