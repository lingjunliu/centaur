
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_logical_xor_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean arrays
    x1 = np.array([True, False, True, False])
    x2 = np.array([False, False, True, True])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Numerical arrays that can be interpreted as booleans
    x1 = np.array([1, 0, 1, 0])
    x2 = np.array([0, 0, 1, 1])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Arrays with different shapes (broadcastable)
    x1 = np.array([[True, False], [False, True]])
    x2 = np.array([False, True])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multidimensional arrays
    x1 = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    x2 = np.array([[[False, True], [True, False]], [[True, False], [False, True]]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Arrays with mixed types (should be implicitly converted)
    x1 = np.array([1, 0, True, False])
    x2 = np.array([0, 1, False, True])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All True
    x1 = np.array([True, True, True, True])
    x2 = np.array([True, True, True, True])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All False
    x1 = np.array([False, False, False, False])
    x2 = np.array([False, False, False, False])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Combination of True, False, 1, 0 with different shapes
    x1 = np.array([[True, 0], [1, False]])
    x2 = np.array([False, 1])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D arrays
    x1 = np.random.choice([True, False], size=(2, 3, 4))
    x2 = np.random.choice([True, False], size=(2, 3, 4))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: arrays with zeros and ones, different shapes for broadcasting
    x1 = np.array([0, 1, 0, 1])
    x2 = np.array([[0], [1], [1], [0]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.logical_xor"] = tf_experimental_numpy_logical_xor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.logical_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.logical_xor'.")

check_valid('tf.experimental.numpy.logical_xor', generated_inputs['tf.experimental.numpy.logical_xor'], lib="tf", suffix=0)
