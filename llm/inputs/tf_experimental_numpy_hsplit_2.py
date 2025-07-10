
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_hsplit_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array, split into 2
    ary = tf.constant(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
    indices_or_sections = [2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, split into 4 equal sections
    ary = tf.constant(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
    indices_or_sections = [4]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, split at multiple indices
    ary = tf.constant(np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))
    indices_or_sections = [2, 4]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, split into 2 sections along the horizontal axis (axis 1)
    ary = tf.constant(np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]))
    indices_or_sections = [2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, split into 4 equal sections along axis 1
    ary = tf.constant(np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]))
    indices_or_sections = [4]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array, should be treated as a 2D array (row vector)
    ary = tf.constant(np.array([1, 2, 3, 4]))
    indices_or_sections = [2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different data type (float)
    ary = tf.constant(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
    indices_or_sections = [2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.hsplit_2"] = tf_experimental_numpy_hsplit_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.hsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.hsplit_2'.")

check_valid('tf.experimental.numpy.hsplit', generated_inputs['tf.experimental.numpy.hsplit_2'], lib="tf", suffix=2)
