
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_imag_inputs():
    list_of_inputs = []

    # Input 1: Simple complex tensor
    val = tf.constant([1+2j, 3+4j, 5+6j], dtype=tf.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex tensor with negative imaginary parts
    val = tf.constant([1-2j, 3-4j, 5-6j], dtype=tf.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor with some zero imaginary parts
    val = tf.constant([1+0j, 3+4j, 5+0j], dtype=tf.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional complex tensor
    val = tf.constant([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=tf.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex tensor with float real and imaginary parts
    val = tf.constant([1.5+2.5j, 3.5+4.5j, 5.5+6.5j], dtype=tf.complex64)
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
