
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_imag_inputs():
    list_of_inputs = []

    # Input 1: complex64, default Tout
    input1 = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    input_dict = {"input": input1, "Tout": tf.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: complex128, Tout=tf.float64
    input2 = np.array([-1 + 2j, -3 + 4j, -5 + 6j], dtype=np.complex128)
    input_dict = {"input": input2, "Tout": tf.float64, "name": "imag_part"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, 2D array
    input3 = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    input_dict = {"input": input3, "Tout": tf.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, 2D array, Tout=tf.float64
    input4 = np.array([[-1 + 2j, -3 + 4j], [-5 + 6j, -7 + 8j]], dtype=np.complex128)
    input_dict = {"input": input4, "Tout": tf.float64, "name": "imag_part"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, 3D array
    input5 = np.array([[[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], [[9 + 10j, 11 + 12j], [13 + 14j, 15 + 16j]]], dtype=np.complex64)
    input_dict = {"input": input5, "Tout": tf.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, 3D array, Tout=tf.float64
    input6 = np.array([[[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], [[9 + 10j, 11 + 12j], [13 + 14j, 15 + 16j]]], dtype=np.complex128)
    input_dict = {"input": input6, "Tout": tf.float64, "name": "imag_part"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64, empty array
    input7 = np.array([], dtype=np.complex64)
    input_dict = {"input": input7, "Tout": tf.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128, empty array, Tout=tf.float64
    input8 = np.array([], dtype=np.complex128)
    input_dict = {"input": input8, "Tout": tf.float64, "name": "imag_part"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: complex64 with zero imaginary part
    input9 = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex64)
    input_dict = {"input": input9, "Tout": tf.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128 with large imaginary part
    input10 = np.array([1 + 1e9j, 2 - 1e9j, 3 + 2e9j], dtype=np.complex128)
    input_dict = {"input": input10, "Tout": tf.float64, "name": "imag_part"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Imag"] = tf_raw_ops_imag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Imag'.")

check_valid('tf.raw_ops.Imag', generated_inputs['tf.raw_ops.Imag'], lib="tf", suffix=0)
