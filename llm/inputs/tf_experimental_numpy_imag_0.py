
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_imag_inputs():
    list_of_inputs = []

    # Input 1: Complex tensor
    val = tf.constant(np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Real tensor (imaginary part is 0)
    val = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multidimensional complex tensor
    val = tf.constant(np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multidimensional real tensor
    val = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex tensor with negative imaginary parts
    val = tf.constant(np.array([1-2j, 3-4j, 5-6j], dtype=np.complex128))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex tensor with mixed signs
    val = tf.constant(np.array([1+2j, 3-4j, -5+6j], dtype=np.complex128))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex tensor with zeros
    val = tf.constant(np.array([0+0j, 1+0j, 0+1j], dtype=np.complex128))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multidimensional complex tensor with mixed signs and zeros
    val = tf.constant(np.array([[1+2j, 0-0j], [-3+0j, 4-5j]], dtype=np.complex128))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank 3 complex tensor
    val = tf.constant(np.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]], dtype=np.complex128))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex tensor with large values
    val = tf.constant(np.array([1e9+2e9j, 3e9+4e9j], dtype=np.complex128))
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.imag"] = tf_experimental_numpy_imag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.imag'.")

check_valid('tf.experimental.numpy.imag', generated_inputs['tf.experimental.numpy.imag'], lib="tf", suffix=0)
