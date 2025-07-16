
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_concatenate_inputs():
    list_of_inputs = []

    # Input 1: Basic concatenation along axis 0
    arys = [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])]
    axis = 0
    input_dict = {'arys': arys, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Concatenation along axis 1
    arys = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = 1
    input_dict = {'arys': arys, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Concatenation of 3 arrays
    arys = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    axis = 0
    input_dict = {'arys': arys, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Concatenation of arrays with different shapes (but compatible)
    arys = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]
    axis = 0
    input_dict = {'arys': arys, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Concatenation with a negative axis
    arys = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = -1
    input_dict = {'arys': arys, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Concatenation of 1D arrays
    arys = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    axis = 0
    input_dict = {'arys': arys, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Concatenation of arrays with different dtypes, resulting in upcasting
    arys = [np.array([1, 2], dtype=np.int32), np.array([3.0, 4.0], dtype=np.float32)]
    axis = 0
    input_dict = {'arys': arys, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays
    arys = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]]])]
    axis = 0
    input_dict = {'arys': arys, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D arrays, axis = 1
    arys = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    axis = 1
    input_dict = {'arys': arys, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty arrays with compatible shapes
    arys = [np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)]
    axis = 0
    input_dict = {'arys': arys, 'axis': axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.concatenate"] = tf_experimental_numpy_concatenate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.concatenate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.concatenate'.")

check_valid('tf.experimental.numpy.concatenate', generated_inputs['tf.experimental.numpy.concatenate'], lib="tf", suffix=0)
