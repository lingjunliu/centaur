
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def generate_bessel_j1_inputs():
    list_of_inputs = []
    
    # Input 1: Positive float32 values
    x = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Negative float32 values
    x = np.array([-0.5, -1.0, -2.0, -4.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Mixed positive and negative float32 values
    x = np.array([0.5, -1.0, 2.0, -4.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Single element array
    x = np.array([1.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Zero values
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Float64 values
    x = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large values
    x = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Small values
    x = np.array([0.001, 0.0001, 0.00001], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Complex array (1D)
    x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Multiple dimensions
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.special.bessel_j1"] = generate_bessel_j1_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.special.bessel_j1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_j1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.special.bessel_j1', generated_inputs['tf.math.special.bessel_j1'], lib="tf", suffix=0)
