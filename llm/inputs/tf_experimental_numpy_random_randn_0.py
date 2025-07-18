
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_experimental_numpy_random_randn_inputs():
    list_of_inputs = []

    # Input 1: 2-D matrix
    input_dict = {
        'args': (2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3-D tensor
    input_dict = {
        'args': (4, 2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Square 2-D matrix
    input_dict = {
        'args': (5, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4-D tensor
    input_dict = {
        'args': (1, 2, 3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Shape with a dimension of 1
    input_dict = {
        'args': (1, 8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another 3-D tensor
    input_dict = {
        'args': (3, 1, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: a larger 2-D matrix
    input_dict = {
        'args': (10, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: another 4-D tensor
    input_dict = {
        'args': (2, 1, 2, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: A different 2-D shape
    input_dict = {
        'args': (6, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A different 3-D shape
    input_dict = {
        'args': (3, 3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.random.randn"] = get_tf_experimental_numpy_random_randn_inputs()

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
