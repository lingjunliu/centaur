
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_subtract_inputs():
    list_of_inputs = []

    # Input 1: Basic subtraction of two scalars
    x1 = tf.constant(5, dtype=tf.int32)
    x2 = tf.constant(2, dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Subtraction of two 1D tensors
    x1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    x2 = tf.constant([5, 4, 3, 2, 1], dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Subtraction of two 2D tensors
    x1 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    x2 = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Subtraction with negative values
    x1 = tf.constant([-1, -2, -3], dtype=tf.int32)
    x2 = tf.constant([1, 2, 3], dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Subtraction of two 3D tensors
    x1 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float64)
    x2 = tf.constant([[[8, 7], [6, 5]], [[4, 3], [2, 1]]], dtype=tf.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Subtraction with different data types (float64)
    x1 = tf.constant([1.5, 2.5, 3.5], dtype=tf.float64)
    x2 = tf.constant([0.5, 1.5, 2.5], dtype=tf.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Subtraction with zeros
    x1 = tf.constant([0, 0, 0], dtype=tf.int32)
    x2 = tf.constant([1, 2, 3], dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger tensor sizes
    x1 = tf.constant(np.random.rand(10, 10), dtype=tf.float32)
    x2 = tf.constant(np.random.rand(10, 10), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Subtraction with broadcasting (x2 is a scalar)
    x1 = tf.constant([1, 2, 3], dtype=tf.int32)
    x2 = tf.constant(1, dtype=tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex shapes
    x1 = tf.constant(np.random.rand(2, 3, 4, 5), dtype=tf.float32)
    x2 = tf.constant(np.random.rand(2, 3, 4, 5), dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    for i in range(len(list_of_inputs)):
        for key in list_of_inputs[i].keys():
            if isinstance(list_of_inputs[i][key], tf.Tensor):
                list_of_inputs[i][key] = list_of_inputs[i][key].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.subtract"] = tf_experimental_numpy_subtract_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.subtract' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.subtract'.")

check_valid('tf.experimental.numpy.subtract', generated_inputs['tf.experimental.numpy.subtract'], lib="tf", suffix=0)
