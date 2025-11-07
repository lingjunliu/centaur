
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ravel_inputs():
    list_of_inputs = []
    
    # Input 1: 2D array
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D array
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D array
    a = np.array([1, 2, 3, 4, 5])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D array (corrected)
    a = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[9, 0], [1, 2]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single element array
    a = np.array([[[[1]]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Negative values
    a = np.array([[-1, -2, -3], [-4, -5, -6]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Float array
    a = np.array([[1.5, 2.5], [3.5, 4.5]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Zero values
    a = np.array([[0, 1], [0, 2]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Mixed values
    a = np.array([[-1, 0, 1], [2, -3, 4]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Large numbers
    a = np.array([[1000000, 2000000], [3000000, 4000000]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.ravel"] = tf_experimental_numpy_ravel_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.ravel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ravel'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.ravel', generated_inputs['tf.experimental.numpy.ravel'], lib="tf", suffix=0)
