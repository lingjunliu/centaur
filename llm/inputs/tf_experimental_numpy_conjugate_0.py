
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_conjugate_inputs():
    list_of_inputs = []

    # Input 1: Real tensor
    x = tf.constant(np.array([1, 2, 3]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex tensor
    x = tf.constant(np.array([1+1j, 2+2j, 3+3j]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multidimensional complex tensor
    x = tf.constant(np.array([[1+1j, 2+2j], [3+3j, 4+4j]]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Real tensor with negative values
    x = tf.constant(np.array([-1, -2, -3]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex tensor with negative real and imaginary parts
    x = tf.constant(np.array([-1-1j, -2-2j, -3-3j]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with mixed positive and negative complex numbers
    x = tf.constant(np.array([1-1j, -2+2j, 3+0j]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher dimensional complex tensor
    x = tf.constant(np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar
    x = tf.constant(1+1j)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty tensor
    x = tf.constant(np.array([]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large values
    x = tf.constant(np.array([1e9+1e9j, 2e9+2e9j]))
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.conjugate"] = tf_experimental_numpy_conjugate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.conjugate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.conjugate'.")

check_valid('tf.experimental.numpy.conjugate', generated_inputs['tf.experimental.numpy.conjugate'], lib="tf", suffix=0)
