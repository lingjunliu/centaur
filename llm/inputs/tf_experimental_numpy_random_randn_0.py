
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_random_randn_inputs():
    list_of_inputs = []

    # Input 1: 1D array
    input_dict = {'args': (3,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array
    input_dict = {'args': (2, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array
    input_dict = {'args': (2, 3, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D array
    input_dict = {'args': (1, 4, 2, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger 1D array
    input_dict = {'args': (10,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger 2D array
    input_dict = {'args': (5, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with a dimension of size 1
    input_dict = {'args': (1, 8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array with a dimension of size 1
    input_dict = {'args': (7, 1, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array
    input_dict = {'args': (1, 1, 1, 1, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another 1D array
    input_dict = {'args': (6,)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: A different 2D array
    input_dict = {'args': (9, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: A different 3D array
    input_dict = {'args': (3, 1, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.random.randn"] = tf_experimental_numpy_random_randn_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.random.randn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.random.randn'.")

check_valid('tf.experimental.numpy.random.randn', generated_inputs['tf.experimental.numpy.random.randn'], lib="tf", suffix=0)
