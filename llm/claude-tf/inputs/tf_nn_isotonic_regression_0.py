
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_nn_isotonic_regression_inputs():
    list_of_inputs = []
    
    input_dict = {
        "inputs": np.array([[3, 1, 2], [1, 3, 4]], dtype=np.float32),
        "decreasing": True,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inputs": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "decreasing": False,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inputs": np.array([5, 3, 7, 2, 9], dtype=np.float32),
        "decreasing": True,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inputs": np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32),
        "decreasing": False,
        "axis": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inputs": np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float32),
        "decreasing": True,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inputs": np.array([[5, -3, 2], [-1, 4, -2]], dtype=np.float32),
        "decreasing": False,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inputs": np.array([42.0], dtype=np.float32),
        "decreasing": True,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inputs": np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float32),
        "decreasing": False,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inputs": np.array([[10.5, 5.5, 7.5]], dtype=np.float32),
        "decreasing": True,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inputs": np.array([[9, 7, 5, 3, 1]], dtype=np.float32),
        "decreasing": True,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.isotonic_regression"] = tf_nn_isotonic_regression_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.isotonic_regression' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.isotonic_regression'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.isotonic_regression', generated_inputs['tf.nn.isotonic_regression'], lib="tf", suffix=0)
