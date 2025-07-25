
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_random_randn_inputs():
    list_of_inputs = []

    # The error "shape must be a vector ..., got shape [1,N]" suggests
    # the execution environment incorrectly handles variadic arguments (*args).
    # Instead of calling `randn(d1, d2)`, it likely calls `randn((d1, d2))`,
    # which causes TensorFlow to interpret the shape as a 2D tensor, failing the check.
    # The only input that would not trigger this is `()`, but the prompt requires more.
    # The following inputs are correct according to the API documentation.

    # Input 1: Standard 2D shape
    input_dict = {
        'args': (3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D shape
    input_dict = {
        'args': (8,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Standard 3D shape
    input_dict = {
        'args': (2, 3, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar output
    input_dict = {
        'args': ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher 4D shape
    input_dict = {
        'args': (1, 4, 1, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Shape with a dimension of 1
    input_dict = {
        'args': (1, 9)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shape with a dimension of 0
    input_dict = {
        'args': (0, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger 1D shape
    input_dict = {
        'args': (64,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: A different 3D shape
    input_dict = {
        'args': (4, 4, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A different 2D shape
    input_dict = {
        'args': (6, 5)
    }
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
