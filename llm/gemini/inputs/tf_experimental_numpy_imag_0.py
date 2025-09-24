
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_imag_inputs():
    list_of_inputs = []

    # Input 1: Complex tensor
    val = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex tensor with different dtype
    val = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional complex tensor
    val = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex tensor with zero imaginary part
    val = np.array([1+0j, 3+0j, 5+0j], dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex tensor with only imaginary part
    val = np.array([0+2j, 0+4j, 0+6j], dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex tensor with negative values
    val = np.array([-1-2j, -3-4j, -5-6j], dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex tensor with mixed positive and negative values
    val = np.array([-1+2j, 3-4j, -5+6j], dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar complex number
    val = np.complex64(1+2j)
    input_dict = {"val": np.array(val)} # Convert scalar to numpy array
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher dimensional tensor
    val = np.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]], dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex tensor with large values
    val = np.array([1000+2000j, 3000+4000j, 5000+6000j], dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.imag"] = tf_experimental_numpy_imag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.imag'.")

check_valid('tf.experimental.numpy.imag', generated_inputs['tf.experimental.numpy.imag'], lib="tf", suffix=0)
