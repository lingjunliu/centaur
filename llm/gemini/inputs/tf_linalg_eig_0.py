
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_eig_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 matrix
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "eig1"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 2x2 matrices
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "eig2"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3x3 matrix
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    name = "eig3"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex matrix
    tensor = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], dtype=np.complex64)
    name = "eig4"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch of complex matrices
    tensor = np.array([[[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], [[5.0, 6.0 + 1j], [7.0 - 1j, 8.0]]], dtype=np.complex64)
    name = "eig5"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger matrix (4x4)
    tensor = np.random.rand(4, 4).astype(np.float32)
    name = "eig6"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch of larger matrices (3x4x4)
    tensor = np.random.rand(3, 4, 4).astype(np.float32)
    name = "eig7"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex, larger matrix (4x4)
    tensor = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    tensor = tensor.astype(np.complex64)
    name = "eig8"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex, Batch of larger matrices (2x3x3)
    tensor = np.random.rand(2, 3, 3) + 1j * np.random.rand(2, 3, 3)
    tensor = tensor.astype(np.complex64)
    name = "eig9"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All zeros
    tensor = np.zeros((3, 3), dtype=np.float32)
    name = "eig10"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.eig"] = tf_linalg_eig_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.eig' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.eig'.")

check_valid('tf.linalg.eig', generated_inputs['tf.linalg.eig'], lib="tf", suffix=0)
