
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_stack_inputs():
    list_of_inputs = []

    # Input 1
    arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis = 1
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    arrays = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    axis = -1
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    arrays = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    axis = 2
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    arrays = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    arrays = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]
    axis = 1
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    arrays = [np.array([1]), np.array([2]), np.array([3])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    arrays = [np.array([[1]]), np.array([[2]]), np.array([[3]])]
    axis = 0
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    arrays = [np.array([1,2,3,4]), np.array([5,6,7,8])]
    axis = -2
    input_dict = {"arrays": arrays, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp = tf_experimental_numpy_stack_inputs()
for i in range(len(temp)):
    generated_inputs["tf.experimental.numpy.stack"] = [ {k: np.array(v) if k == 'arrays' else v for k, v in temp[i].items()}]
    break

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.stack'.")

check_valid('tf.experimental.numpy.stack', generated_inputs['tf.experimental.numpy.stack'], lib="tf", suffix=0)
