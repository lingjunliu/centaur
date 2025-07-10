
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_ravel_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    a = tf.constant([1, 2, 3])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty tensor
    a = tf.constant([])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with different data type (float)
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with different data type (int64)
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.int64)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with negative values
    a = tf.constant([[-1, 2], [3, -4]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor
    a = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar tensor
    a = tf.constant(5)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: tensor with 0
    a = tf.constant([[0, 1], [2, 0]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.ravel"] = tf_experimental_numpy_ravel_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.ravel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ravel'.")

check_valid('tf.experimental.numpy.ravel', generated_inputs['tf.experimental.numpy.ravel'], lib="tf", suffix=0)
