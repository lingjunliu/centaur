
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_pad_inputs():
    list_of_inputs = []

    # Input 1: 1D array, constant mode
    array = np.array([1, 2, 3])
    pad_width = [[1, 1]]
    mode = 'constant'
    input_dict = {"array": array, "pad_width": pad_width, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, reflect mode
    array = np.array([[1, 2], [3, 4]])
    pad_width = [[1, 1], [1, 1]]
    mode = 'reflect'
    input_dict = {"array": array, "pad_width": pad_width, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, symmetric mode
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    pad_width = [[1, 1], [0, 1], [1, 0]]
    mode = 'symmetric'
    input_dict = {"array": array, "pad_width": pad_width, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different padding on different sides, constant mode
    array = np.array([[1, 2], [3, 4]])
    pad_width = [[2, 0], [0, 2]]
    mode = 'constant'
    input_dict = {"array": array, "pad_width": pad_width, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero padding, reflect mode
    array = np.array([1, 2, 3])
    pad_width = [[0, 0]]
    mode = 'reflect'
    input_dict = {"array": array, "pad_width": pad_width, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: Single value pad_width, symmetric mode
    array = np.array([[1, 2], [3, 4]])
    pad_width = [[1, 1], [1, 1]]
    mode = 'symmetric'
    input_dict = {"array": array, "pad_width": pad_width, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  constant mode with different pad widths
    array = np.array([1, 2, 3, 4])
    pad_width = [[2, 1]]
    mode = 'constant'
    input_dict = {"array": array, "pad_width": pad_width, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: reflect mode, different axis
    array = np.array([[1, 2, 3], [4, 5, 6]])
    pad_width = [[0, 0], [1, 1]]
    mode = 'reflect'
    input_dict = {"array": array, "pad_width": pad_width, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: symmetric mode, valid padding
    array = np.array([1, 2])
    pad_width = [[1, 1]]
    mode = 'symmetric'
    input_dict = {"array": array, "pad_width": pad_width, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: constant mode, 3D with asymmetric padding
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    pad_width = [[0, 1], [0, 0], [0, 0]]
    mode = 'constant'
    input_dict = {"array": array, "pad_width": pad_width, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.pad"] = tf_experimental_numpy_pad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.pad'.")

check_valid('tf.experimental.numpy.pad', generated_inputs['tf.experimental.numpy.pad'], lib="tf", suffix=0)
