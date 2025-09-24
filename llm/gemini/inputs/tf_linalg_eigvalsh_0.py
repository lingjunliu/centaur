
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_eigvalsh_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 matrix
    tensor = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 matrix
    tensor = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": "eigvalsh_3x3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of 2x2 matrices
    tensor = np.array([[[1.0, 0.0], [0.0, 2.0]], [[3.0, 0.0], [0.0, 4.0]]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger matrix with negative values
    tensor = np.array([[4.0, -1.0, 2.0], [-1.0, 5.0, -3.0], [2.0, -3.0, 6.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": "eigvalsh_neg"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex matrix (still self-adjoint if conjugate transpose equals itself)
    tensor = np.array([[1+0j, 2-1j], [2+1j, 3+0j]], dtype=np.complex64)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch of complex matrices
    tensor = np.array([[[1+0j, 2-1j], [2+1j, 3+0j]], [[4+0j, 1-2j], [1+2j, 5+0j]]], dtype=np.complex64)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element matrix
    tensor = np.array([[5.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": "eigvalsh_single"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batch of single element matrices
    tensor = np.array([[[5.0]], [[6.0]]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another 3x3 matrix
    tensor = np.array([[7.0, 2.0, 1.0], [2.0, 8.0, 3.0], [1.0, 3.0, 9.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batch of matrices with different sizes
    tensor = np.array([[[1.0, 2.0], [2.0, 3.0]], [[4.0]]], dtype='object')

    tensor = np.array([[[1.0, 2.0], [2.0, 3.0]], [[4.0, 0.0], [0.0, 4.0]]], dtype=np.float32)

    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Add another valid input with different values to exceed the 10 input limit
    tensor = np.array([[10.0, 1.0], [1.0, 11.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": "eigvalsh_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.eigvalsh"] = tf_linalg_eigvalsh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.eigvalsh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.eigvalsh'.")

check_valid('tf.linalg.eigvalsh', generated_inputs['tf.linalg.eigvalsh'], lib="tf", suffix=0)
