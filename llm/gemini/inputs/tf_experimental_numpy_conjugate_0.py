
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_conjugate_inputs():
    list_of_inputs = []

    # Input 1: Real numbers
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex numbers
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative complex numbers
    x = np.array([-1 - 1j, -2 - 2j, -3 - 3j], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero complex numbers
    x = np.array([0 + 0j, 0 + 0j, 0 + 0j], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of complex numbers
    x = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of complex numbers
    x = np.array([[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], [[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]]], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed positive and negative complex numbers
    x = np.array([1 - 1j, -2 + 2j, 3 - 3j], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Real numbers with float64 dtype
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex numbers with complex128 dtype
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with only imaginary components.
    x = np.array([1j, 2j, 3j], dtype=np.complex64)
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
