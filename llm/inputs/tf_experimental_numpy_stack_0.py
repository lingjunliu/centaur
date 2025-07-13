
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_stack_inputs():
    list_of_inputs = []

    # Input 1: Simple stack along axis 0
    arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Stack along axis 1
    arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    axis = 1
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Stack with different shapes that are stackable
    arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.stack"] = tf_experimental_numpy_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.stack'.")

check_valid('tf.experimental.numpy.stack', generated_inputs['tf.experimental.numpy.stack'], lib="tf", suffix=0)
