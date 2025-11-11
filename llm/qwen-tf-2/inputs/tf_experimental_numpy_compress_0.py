
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experiment_numpy_compress_inputs():
    list_of_inputs = []

    
    # Input 1, valid
    condition = np.array([True, False, True])
    a = np.array([1, 2, 3])
    axis = None
    
    input_dict = {
        "condition": condition,
        "a": a,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    condition = np.array([True, False, True, False])
    a = np.array([[1, 2], [3, 4], [5, 6]])
    axis = 0
    
    input_dict = {
        "condition": condition,
        "a": a,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    condition = np.array([True, False, True])
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 1
    
    input_dict = {
        "condition": condition,
        "a": a,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    condition = np.array([False, False, True, True])
    a = np.array([1, 2, 3, 4])
    axis = None
    
    input_dict = {
        "condition": condition,
        "a": a,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    condition = np.array([True, True, False, False])
    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    axis = 0
    
    input_dict = {
        "condition": condition,
        "a": a,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    condition = np.array([True, False])
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    axis = 1
    
    input_dict = {
        "condition": condition,
        "a": a,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    condition = np.array([False, True, False])
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 0
    
    input_dict = {
        "condition": condition,
        "a": a,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    condition = np.array([True, False, True])
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    axis = 0
    
    input_dict = {
        "condition": condition,
        "a": a,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    condition = np.array([False, True, False, True])
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    axis = 1
    
    input_dict = {
        "condition": condition,
        "a": a,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    condition = np.array([True, False, True, False])
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 1
    
    input_dict = {
        "condition": condition,
        "a": a,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.compress"] = tf_experiment_numpy_compress_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.compress' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.compress'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.compress', generated_inputs['tf.experimental.numpy.compress'], lib="tf", suffix=0)
