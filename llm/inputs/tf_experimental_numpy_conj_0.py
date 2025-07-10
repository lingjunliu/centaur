
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_conj_inputs():
    list_of_inputs = []

    # Input 1: Simple complex tensor
    x = tf.constant([1 + 1j, 2 + 2j, 3 + 3j], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex tensor with negative values
    x = tf.constant([-1 - 1j, -2 + 2j, 3 - 3j], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Real tensor (should still work, no change)
    x = tf.constant([1, 2, 3], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional complex tensor
    x = tf.constant([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional real tensor
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex128 tensor
    x = tf.constant([1 + 1j, 2 + 2j, 3 + 3j], dtype=tf.complex128)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero tensor
    x = tf.constant([0 + 0j, 0 + 0j], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with only real part
    x = tf.constant([1 + 0j, 2 + 0j, 3 + 0j], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with only imaginary part
    x = tf.constant([0 + 1j, 0 + 2j, 0 + 3j], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex tensor with mixed positive and negative real/imaginary parts
    x = tf.constant([-1 + 1j, 2 - 2j, -3 - 3j, 4 + 0j, 0 - 5j], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty Tensor - removing this because it may cause issues

    # Input 12: Tensor with different shapes
    x = tf.constant([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Convert to numpy arrays
    for input_dict in list_of_inputs:
        input_dict['x'] = input_dict['x'].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.conj"] = tf_experimental_numpy_conj_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.conj' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.conj'.")

check_valid('tf.experimental.numpy.conj', generated_inputs['tf.experimental.numpy.conj'], lib="tf", suffix=0)
