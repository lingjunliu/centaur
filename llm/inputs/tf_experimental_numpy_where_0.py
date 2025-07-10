
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_where_inputs():
    list_of_inputs = []

    # Input 1
    condition = tf.constant([True, False, True])
    x = tf.constant([1, 2, 3]).numpy()
    y = tf.constant([4, 5, 6]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    condition = tf.constant([[True, False], [False, True]])
    x = tf.constant([[1, 2], [3, 4]]).numpy()
    y = tf.constant([[5, 6], [7, 8]]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    condition = tf.constant([[[True, False], [False, True]], [[False, True], [True, False]]])
    x = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    y = tf.constant([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    condition = tf.constant([False, False, False])
    x = tf.constant([1, 2, 3]).numpy()
    y = tf.constant([4, 5, 6]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    condition = tf.constant([True, True, True])
    x = tf.constant([1, 2, 3]).numpy()
    y = tf.constant([4, 5, 6]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - different values
    condition = tf.constant([True, False, True])
    x = tf.constant([-1, -2, -3]).numpy()
    y = tf.constant([-4, -5, -6]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - mixed values
    condition = tf.constant([True, False, True])
    x = tf.constant([-1, 2, -3]).numpy()
    y = tf.constant([4, -5, 6]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - all false multi dim
    condition = tf.constant([[False, False], [False, False]])
    x = tf.constant([[1, 2], [3, 4]]).numpy()
    y = tf.constant([[5, 6], [7, 8]]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - all true multi dim
    condition = tf.constant([[True, True], [True, True]])
    x = tf.constant([[1, 2], [3, 4]]).numpy()
    y = tf.constant([[5, 6], [7, 8]]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - different shapes but broadcastable condition
    condition = tf.constant([True, False])
    x = tf.constant([[1, 2], [3, 4]]).numpy()
    y = tf.constant([[5, 6], [7, 8]]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11 - 1D tensors
    condition = tf.constant([True, False, True, False])
    x = tf.constant([1, 2, 3, 4]).numpy()
    y = tf.constant([5, 6, 7, 8]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12 - 2D tensors
    condition = tf.constant([[True, False], [True, False]])
    x = tf.constant([[1, 2], [3, 4]]).numpy()
    y = tf.constant([[5, 6], [7, 8]]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13 - float tensors
    condition = tf.constant([True, False, True])
    x = tf.cast(tf.constant([1.0, 2.0, 3.0]), dtype=tf.float32).numpy()
    y = tf.cast(tf.constant([4.0, 5.0, 6.0]), dtype=tf.float32).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14 - bool tensors
    condition = tf.constant([True, False, True])
    x = tf.constant([True, False, True]).numpy()
    y = tf.constant([False, True, False]).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 15 - Mixed types
    condition = tf.constant([True, False, True])
    x = tf.cast(tf.constant([1, 2, 3]), dtype=tf.int32).numpy()
    y = tf.cast(tf.constant([4.0, 5.0, 6.0]), dtype=tf.float32).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 16 - int64
    condition = tf.constant([True, False, True])
    x = tf.cast(tf.constant([1, 2, 3]), dtype=tf.int64).numpy()
    y = tf.cast(tf.constant([4, 5, 6]), dtype=tf.int64).numpy()
    condition = condition.numpy()
    input_dict = {"condition": condition, "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.where"] = tf_experimental_numpy_where_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.where' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.where'.")

check_valid('tf.experimental.numpy.where', generated_inputs['tf.experimental.numpy.where'], lib="tf", suffix=0)
