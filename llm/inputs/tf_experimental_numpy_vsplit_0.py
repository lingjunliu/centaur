
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_vsplit_inputs():
    list_of_inputs = []

    # Input 1: 2D array, split into 2
    ary = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    indices_or_sections = 2
    input_dict = {'ary': ary, 'indices_or_sections': indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, split into 4
    ary = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    indices_or_sections = 4
    input_dict = {'ary': ary, 'indices_or_sections': indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, split into 2 along axis 0
    ary = np.arange(24).reshape((6, 2, 2))
    indices_or_sections = 2
    input_dict = {'ary': ary, 'indices_or_sections': indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, split into 3 along axis 0
    ary = np.arange(24).reshape((6, 2, 2))
    indices_or_sections = 3
    input_dict = {'ary': ary, 'indices_or_sections': indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, split into 3
    ary = np.array([[1, 2], [3, 4], [5, 6]])
    indices_or_sections = 3
    input_dict = {'ary': ary, 'indices_or_sections': indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, split into 1
    ary = np.array([[1, 2], [3, 4], [5, 6]])
    indices_or_sections = 1
    input_dict = {'ary': ary, 'indices_or_sections': indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, split into 3
    ary = np.array([1, 2, 3, 4, 5, 6])
    ary = ary.reshape((6,1))
    indices_or_sections = 3
    input_dict = {'ary': ary, 'indices_or_sections': indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, split into 6
    ary = np.array([1, 2, 3, 4, 5, 6])
    ary = ary.reshape((6,1))
    indices_or_sections = 6
    input_dict = {'ary': ary, 'indices_or_sections': indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, split into 5
    ary = np.arange(10).reshape((10,1))
    indices_or_sections = 5
    input_dict = {'ary': ary, 'indices_or_sections': indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D array, split into 2 along axis 0
    ary = np.arange(32).reshape((4, 2, 2, 2))
    indices_or_sections = 2
    input_dict = {'ary': ary, 'indices_or_sections': indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.vsplit"] = tf_experimental_numpy_vsplit_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.vsplit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.vsplit'.")

check_valid('tf.experimental.numpy.vsplit', generated_inputs['tf.experimental.numpy.vsplit'], lib="tf", suffix=0)
