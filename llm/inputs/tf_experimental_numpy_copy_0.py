
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_numpy_copy_inputs():
    list_of_inputs = []
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    # Input 1: Basic 1D tensor
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with different data type
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int64)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with negative values
    a = tf.constant([-1, -2, 0, 1, 2], dtype=tf.int32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty tensor
    a = tf.constant([], dtype=tf.int32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with boolean values
    a = tf.constant([True, False, True], dtype=tf.bool)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with all zeros
    a = tf.zeros((3, 3), dtype=tf.int32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with all ones
    a = tf.ones((2, 2), dtype=tf.float32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher dimension tensor
    a = tf.constant(np.random.rand(2, 3, 4, 5), dtype=tf.float32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.copy"] = tf_experimental_numpy_copy_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.copy'.")

check_valid('tf.experimental.numpy.copy', generated_inputs['tf.experimental.numpy.copy'], lib="tf", suffix=0)
