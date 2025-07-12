
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic equal arrays
    x1 = np.array([1, 2, 3])
    x2 = np.array([1, 2, 3])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different arrays
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Arrays with different shapes (broadcasting should happen)
    x1 = np.array([[1, 2, 3], [4, 5, 6]])
    x2 = np.array([1, 2, 3])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Arrays with different data types
    x1 = np.array([1, 2, 3], dtype=np.int32)
    x2 = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Arrays with boolean values
    x1 = np.array([True, False, True])
    x2 = np.array([True, True, False])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional arrays
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    x2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Arrays with negative values
    x1 = np.array([-1, -2, -3])
    x2 = np.array([-1, -2, -3])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Arrays with zero values
    x1 = np.array([0, 0, 0])
    x2 = np.array([0, 0, 0])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Arrays with mixed positive and negative values
    x1 = np.array([-1, 2, -3])
    x2 = np.array([-1, 2, -3])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Arrays with different types, but same content after casting (int and float).
    x1 = np.array([1, 2, 3], dtype=np.int32)
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.equal"] = tf_experimental_numpy_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.equal'.")

check_valid('tf.experimental.numpy.equal', generated_inputs['tf.experimental.numpy.equal'], lib="tf", suffix=0)
