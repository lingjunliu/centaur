
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_std_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant(np.array([1, 2, 3, 4, 5]))
    axis = 0
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    axis = 0
    keepdims = True
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    axis = 1
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    axis = 0
    keepdims = True
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    axis = 1
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    axis = 2
    keepdims = True
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    axis = 0
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = tf.constant(np.array([[-1, 2], [3, -4]]))
    axis = 0
    keepdims = True
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = tf.constant(np.array([[-1, 2], [3, -4]]))
    axis = 1
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = tf.constant(np.array([[[1, -2], [3, 4]], [[-5, 6], [7, 8]]]))
    axis = 2
    keepdims = False
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.std"] = tf_experimental_numpy_std_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.std' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.std'.")

check_valid('tf.experimental.numpy.std', generated_inputs['tf.experimental.numpy.std'], lib="tf", suffix=0)
