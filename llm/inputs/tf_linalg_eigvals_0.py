
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_eigvals_inputs():
    list_of_inputs = []

    # Input 1: 2x2 matrix
    tensor = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    name = "eigvals_1"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 2x2 matrices
    tensor = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    name = "eigvals_2"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex 2x2 matrix
    tensor = np.array([[1. + 1j, 2. + 2j], [3. + 3j, 4. + 4j]], dtype=np.complex64)
    name = "eigvals_3"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of complex matrices
    tensor = np.array([[[1. + 0j, 2. + 0j], [3. + 0j, 4. + 0j]], [[5. + 0j, 6. + 0j], [7. + 0j, 8. + 0j]]], dtype=np.complex64)
    name = "eigvals_4"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3x3 matrix
    tensor = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    name = "eigvals_5"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch of 3x3 matrices
    tensor = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], [[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]], dtype=np.float32)
    name = "eigvals_6"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex 3x3 matrix
    tensor = np.array([[1. + 1j, 2. + 2j, 3. + 3j], [4. + 4j, 5. + 5j, 6. + 6j], [7. + 7j, 8. + 8j, 9. + 9j]], dtype=np.complex64)
    name = "eigvals_7"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Matrix with negative values
    tensor = np.array([[-1., 2.], [3., -4.]], dtype=np.float32)
    name = "eigvals_8"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Identity matrix
    tensor = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    name = "eigvals_9"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large values
    tensor = np.array([[1000., 2000.], [3000., 4000.]], dtype=np.float32)
    name = "eigvals_10"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.eigvals"] = tf_linalg_eigvals_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.eigvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.eigvals'.")

check_valid('tf.linalg.eigvals', generated_inputs['tf.linalg.eigvals'], lib="tf", suffix=0)
