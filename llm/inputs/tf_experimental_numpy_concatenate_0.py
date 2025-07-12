
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_concatenate_inputs():
    list_of_inputs = []

    # Input 1
    arys = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    axis = 0
    input_dict = {"arys": arys, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    arys = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = 1
    input_dict = {"arys": arys, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    arys = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    axis = 2
    input_dict = {"arys": arys, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    arys = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    axis = 0
    input_dict = {"arys": arys, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    arys = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
    axis = 0
    input_dict = {"arys": arys, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    arys = [np.array([1]), np.array([2])]
    axis = 0
    input_dict = {"arys": arys, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    arys = [np.array([[1]]), np.array([[2]])]
    axis = 0
    input_dict = {"arys": arys, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    arys = [np.array([1, 2]), np.array([3, 4])]
    axis = 0
    input_dict = {"arys": arys, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    arys = [np.array([[1, 2]]), np.array([[3, 4]])]
    axis = 1
    input_dict = {"arys": arys, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    arys = [np.array([[[1, 2]]]), np.array([[[3, 4]]])]
    axis = 2
    input_dict = {"arys": arys, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_experimental_numpy_concatenate_inputs()
transformed_inputs = []
for input_dict in inputs:
    arys = input_dict['arys']
    arys_transformed = [tf.convert_to_tensor(arr, dtype=tf.float32) for arr in arys]
    transformed_input_dict = {"arys": arys_transformed, "axis": input_dict['axis']}
    transformed_inputs.append(transformed_input_dict)

generated_inputs["tf.experimental.numpy.concatenate"] = transformed_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.concatenate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.concatenate'.")

check_valid('tf.experimental.numpy.concatenate', generated_inputs['tf.experimental.numpy.concatenate'], lib="tf", suffix=0)
