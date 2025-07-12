
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_hsplit_inputs():
    list_of_inputs = []

    # Input 1: 2D array, equal sections
    ary = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    indices_or_sections = [2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, different sections
    ary = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    indices_or_sections = [2, 5]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, equal sections
    ary = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
    indices_or_sections = [2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, different sections
    ary = np.array([[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]])
    indices_or_sections = [2, 4]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, sections equal to the number of columns
    ary = np.array([[1, 2, 3], [4, 5, 6]])
    indices_or_sections = [1, 2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array, splitting into sections
    ary = np.array([1, 2, 3, 4])
    indices_or_sections = [2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large array, splitting into multiple sections
    ary = np.random.rand(10, 20)
    indices_or_sections = [5, 10, 15]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: array with different data type
    ary = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    indices_or_sections = [2]
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: empty list as indices_or_sections
    ary = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    indices_or_sections = []  # This is a valid input.
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: ary with only one column
    ary = np.array([[1], [2], [3]])
    indices_or_sections = [1]  # Splitting at an impossible location will result in an empty array.
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: splitting into more parts than possible
    ary = np.array([[1,2,3]])
    indices_or_sections = [1,2,3,4] # Should still run, but not produce error
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
