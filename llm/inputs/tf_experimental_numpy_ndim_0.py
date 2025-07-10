
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_numpy_ndim_inputs():
    list_of_inputs = []

    # Input 1: Scalar tensor
    a = tf.constant(5)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor
    a = tf.constant([1, 2, 3])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    a = tf.constant([[1, 2], [3, 4]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with different data type (float)
    a = tf.constant(3.14)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D tensor
    a = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty tensor
    a = tf.constant([])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with negative values
    a = tf.constant([-1, -2, -3])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with mixed positive and negative values
    a = tf.constant([[-1, 2], [3, -4]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with boolean values
    a = tf.constant([True, False, True])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.ndim"] = tf_experimental_numpy_ndim_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.ndim' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.ndim'.")

check_valid('tf.experimental.numpy.ndim', generated_inputs['tf.experimental.numpy.ndim'], lib="tf", suffix=0)
