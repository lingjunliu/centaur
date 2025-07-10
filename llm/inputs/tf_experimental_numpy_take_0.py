
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_take_inputs():
    list_of_inputs = []

    tf.experimental.numpy.experimental_enable_numpy_behavior()

    # Input 1
    a = tf.constant(np.array([4, 3, 5, 7, 6, 8]))
    indices = tf.constant(np.array([0, 1, 4]))
    axis = tf.constant(0)
    mode = tf.constant('clip')
    input_dict = {"a": a, "indices": indices, "axis": axis, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    indices = tf.constant(np.array([0, 1]))
    axis = tf.constant(0)
    mode = tf.constant('clip')
    input_dict = {"a": a, "indices": indices, "axis": axis, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant(np.array([[1, 2], [3, 4]]))
    indices = tf.constant(np.array([0, 1]))
    axis = tf.constant(1)
    mode = tf.constant('clip')
    input_dict = {"a": a, "indices": indices, "axis": axis, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant(np.array([4, 3, 5, 7, 6, 8]))
    indices = tf.constant(np.array([-1, -2]))
    axis = tf.constant(0)
    mode = tf.constant('clip')
    input_dict = {"a": a, "indices": indices, "axis": axis, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
    indices = tf.constant(np.array([0, 1]))
    axis = tf.constant(0)
    mode = tf.constant('clip')
    input_dict = {"a": a, "indices": indices, "axis": axis, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant(np.array([1, 2, 3, 4, 5]))
    indices = tf.constant(np.array([2, 4]))
    axis = tf.constant(0)
    mode = tf.constant('clip')
    input_dict = {"a": a, "indices": indices, "axis": axis, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    indices = tf.constant(np.array([0, 1]))
    axis = tf.constant(0)
    mode = tf.constant('clip')
    input_dict = {"a": a, "indices": indices, "axis": axis, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    indices = tf.constant(np.array([0, 1]))
    axis = tf.constant(1)
    mode = tf.constant('clip')
    input_dict = {"a": a, "indices": indices, "axis": axis, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    indices = tf.constant(np.array([0, 1]))
    axis = tf.constant(2)
    mode = tf.constant('clip')
    input_dict = {"a": a, "indices": indices, "axis": axis, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    indices = tf.constant(np.array([0, 2]))
    axis = tf.constant(0)
    mode = tf.constant('clip')
    input_dict = {"a": a, "indices": indices, "axis": axis, "mode": mode}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.take"] = tf_experimental_numpy_take_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.take' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.take'.")

check_valid('tf.experimental.numpy.take', generated_inputs['tf.experimental.numpy.take'], lib="tf", suffix=0)
