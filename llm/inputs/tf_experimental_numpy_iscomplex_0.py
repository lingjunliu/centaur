
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_iscomplex_inputs():
    list_of_inputs = []

    def to_numpy(tensor):
        return tensor.numpy()

    # Input 1: Simple complex tensor
    x = tf.constant([1 + 1j, 2 + 2j, 3 + 0j], dtype=tf.complex64)
    input_dict = {"x": to_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tensor with only real numbers
    x = tf.constant([1, 2, 3], dtype=tf.float32)
    input_dict = {"x": to_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor with only imaginary numbers
    x = tf.constant([1j, 2j, 3j], dtype=tf.complex128)
    input_dict = {"x": to_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional complex tensor
    x = tf.constant([[1 + 1j, 2 + 2j], [3 + 0j, 4 - 1j]], dtype=tf.complex64)
    input_dict = {"x": to_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional real tensor
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    input_dict = {"x": to_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar complex value
    x = tf.constant(5 + 2j, dtype=tf.complex128)
    input_dict = {"x": to_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scalar real value
    x = tf.constant(5, dtype=tf.float32)
    input_dict = {"x": to_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex tensor with negative values
    x = tf.constant([-1 - 1j, -2 + 2j, 3 - 0j], dtype=tf.complex64)
    input_dict = {"x": to_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Complex tensor with mixed dtypes (casted to complex64)
    x = tf.constant([1 + 1j, 2.0, 3 + 0j], dtype=tf.complex64)
    input_dict = {"x": to_numpy(x)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.iscomplex"] = tf_experimental_numpy_iscomplex_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.iscomplex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.iscomplex'.")

check_valid('tf.experimental.numpy.iscomplex', generated_inputs['tf.experimental.numpy.iscomplex'], lib="tf", suffix=0)
