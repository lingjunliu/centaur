
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_all_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([True, True, True])
    axis = None
    keepdims = False
    input_dict = {"a": tf.constant(a).numpy(), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([True, False, True])
    axis = None
    keepdims = False
    input_dict = {"a": tf.constant(a).numpy(), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[True, True], [False, True]])
    axis = 0
    keepdims = False
    input_dict = {"a": tf.constant(a).numpy(), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[True, True], [False, True]])
    axis = 1
    keepdims = False
    input_dict = {"a": tf.constant(a).numpy(), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[True, True], [False, True]])
    axis = 0
    keepdims = True
    input_dict = {"a": tf.constant(a).numpy(), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[True, True], [False, True]])
    axis = 1
    keepdims = True
    input_dict = {"a": tf.constant(a).numpy(), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[[True, True], [True, True]], [[False, True], [True, True]]])
    axis = 0
    keepdims = False
    input_dict = {"a": tf.constant(a).numpy(), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[[True, True], [True, True]], [[False, True], [True, True]]])
    axis = 1
    keepdims = False
    input_dict = {"a": tf.constant(a).numpy(), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[[True, True], [True, True]], [[False, True], [True, True]]])
    axis = 2
    keepdims = False
    input_dict = {"a": tf.constant(a).numpy(), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([[[True, True], [True, True]], [[False, True], [True, True]]])
    axis = 0
    keepdims = True
    input_dict = {"a": tf.constant(a).numpy(), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.all"] = tf_experimental_numpy_all_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.all' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.all'.")

check_valid('tf.experimental.numpy.all', generated_inputs['tf.experimental.numpy.all'], lib="tf", suffix=0)
