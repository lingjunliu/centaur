
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_where_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with True values
    condition = np.array([[True, False], [True, False]], dtype=bool)
    input_dict = {
        "name": "test1",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor with True values
    condition = np.array([[[True, False], [True, False]], [[False, True], [False, True]]], dtype=bool)
    input_dict = {
        "name": "test2",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor with True values
    condition = np.array([True, False, True, False], dtype=bool)
    input_dict = {
        "name": "test3",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with True values
    condition = np.array([[[[True, False], [False, True]], [[False, True], [True, False]]], [[[False, False], [True, False]], [[True, False], [False, True]]]], dtype=bool)
    input_dict = {
        "name": "test4",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with all True values
    condition = np.array([[True, True], [True, True]], dtype=bool)
    input_dict = {
        "name": "test5",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with all False values
    condition = np.array([[False, False], [False, False]], dtype=bool)
    input_dict = {
        "name": "test6",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with negative values
    condition = np.array([[[1.5, -0.5], [0.0, 0.0]], [[0.0, 0.25], [0.0, 0.75]]], dtype=float)
    input_dict = {
        "name": "test7",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensor with complex values
    condition = np.array([[[1.5+0.j, 0.0+0.j], [0.0+0.5j, 0.0+0.j]], [[0.0+0.j, 0.25+1.5j], [0.0+0.j, 0.75+0.j]]], dtype=complex)
    input_dict = {
        "name": "test8",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with negative values
    condition = np.array([[1.5, -0.5], [0.0, 0.0]], dtype=float)
    input_dict = {
        "name": "test9",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 3D tensor with float values
    condition = np.array([[[1.5, 0.0], [-0.5, 0.0]], [[0.0, 0.25], [0.0, 0.75]]], dtype=float)
    input_dict = {
        "name": "test10",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Where"] = tf_raw_ops_where_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Where' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Where'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Where', generated_inputs['tf.raw_ops.Where'], lib="tf", suffix=0)
