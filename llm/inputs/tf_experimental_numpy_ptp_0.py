
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ptp_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([1, 5, 2, 8])
    axis = None
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1, 5, 2, 8], [3, 9, -1, 0]])
    axis = 0
    keepdims = True
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[1, 5, 2, 8], [3, 9, -1, 0]])
    axis = 1
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[[1, 5], [2, 8]], [[3, 9], [-1, 0]]])
    axis = 0
    keepdims = True
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[[1, 5], [2, 8]], [[3, 9], [-1, 0]]])
    axis = 1
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[[1, 5], [2, 8]], [[3, 9], [-1, 0]]])
    axis = 2
    keepdims = True
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([1, 2, 3, 4, 5])
    axis = None
    keepdims = True
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Example with negative values
    a = np.array([-1, 2, -3, 4, -5])
    axis = None
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[1.5, 2.5], [3.5, 4.5]])
    axis = 0
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 2
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.ptp"] = tf_experimental_numpy_ptp_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.ptp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ptp'.")

check_valid('tf.experimental.numpy.ptp', generated_inputs['tf.experimental.numpy.ptp'], lib="tf", suffix=0)
