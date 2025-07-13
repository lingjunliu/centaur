
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_meshgrid_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 1D arrays
    xi = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Three 1D arrays
    xi = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    input_dict = {"xi": xi}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.meshgrid"] = tf_experimental_numpy_meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.meshgrid'.")

check_valid('tf.experimental.numpy.meshgrid', generated_inputs['tf.experimental.numpy.meshgrid'], lib="tf", suffix=0)
