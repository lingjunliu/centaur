
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_logm_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 complex matrix
    input_matrix = np.array([[1+1j, 2+0j], [0+1j, 3-1j]], dtype=np.complex64)
    input_dict = {"input": tf.convert_to_tensor(input_matrix).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 2x2 complex matrices
    input_matrix = np.array([[[1+0j, 2+1j], [3-1j, 4+0j]], [[5+1j, 6-0j], [7+0j, 8+1j]]], dtype=np.complex64)
    input_dict = {"input": tf.convert_to_tensor(input_matrix).numpy(), "name": "batch_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3x3 complex matrix
    input_matrix = np.array([[1+0j, 2+1j, 3-1j], [4-1j, 5+0j, 6+1j], [7+1j, 8-1j, 9+0j]], dtype=np.complex128)
    input_dict = {"input": tf.convert_to_tensor(input_matrix).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2x2 complex matrix with negative values
    input_matrix = np.array([[-1+1j, -2+0j], [0-1j, -3-1j]], dtype=np.complex64)
    input_dict = {"input": tf.convert_to_tensor(input_matrix).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1x1 complex matrix
    input_matrix = np.array([[1+1j]], dtype=np.complex64)
    input_dict = {"input": tf.convert_to_tensor(input_matrix).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.logm"] = tf_linalg_logm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.logm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.logm'.")

check_valid('tf.linalg.logm', generated_inputs['tf.linalg.logm'], lib="tf", suffix=0)
