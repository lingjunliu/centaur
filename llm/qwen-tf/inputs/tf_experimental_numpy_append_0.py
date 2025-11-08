
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tfexperimental_numpy_append_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    arr = np.array([1, 2, 3])
    values = np.array([4, 5, 6])
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[5, 6], [7, 8]])
    axis = 0
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - single element arrays
    arr = np.array([1])
    values = np.array([2])
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - scalar values
    arr = np.array(1)
    values = np.array(2)
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - float arrays
    arr = np.array([1.1, 2.2, 3.3])
    values = np.array([4.4, 5.5, 6.6])
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - negative values
    arr = np.array([-1, -2, -3])
    values = np.array([-4, -5, -6])
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - same dimension arrays
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[5, 6], [7, 8]])
    axis = 1
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - one dimensional arrays
    arr = np.array([1, 2, 3])
    values = np.array([4, 5, 6])
    axis = 0
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.append"] = tfexperimental_numpy_append_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.append' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.append'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.append', generated_inputs['tf.experimental.numpy.append'], lib="tf", suffix=0)
