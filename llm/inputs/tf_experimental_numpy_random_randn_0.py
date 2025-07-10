
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_random_randn_inputs():
    list_of_inputs = []

    tf.random.set_seed(1)

    # Input 1: Scalar
    input_dict = {"args": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single integer
    input_dict = {"args": (5,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple of integers
    input_dict = {"args": (2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three dimensions
    input_dict = {"args": (2, 3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger dimensions
    input_dict = {"args": (10, 10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: One element array
    input_dict = {"args": (1, 1, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another set of dimensions
    input_dict = {"args": (4, 2, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different dimension lengths
    input_dict = {"args": (7, 1, 9)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Even larger array
    input_dict = {"args": (20, 20)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Four dimensions
    input_dict = {"args": (2, 3, 4, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.random.randn"] = tf_experimental_numpy_random_randn_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.random.randn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.random.randn'.")

check_valid('tf.experimental.numpy.random.randn', generated_inputs['tf.experimental.numpy.random.randn'], lib="tf", suffix=0)
