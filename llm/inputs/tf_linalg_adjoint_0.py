
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_adjoint_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 complex matrix
    matrix = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex64)
    input_dict = {"matrix": tf.convert_to_tensor(matrix), "name": "adjoint_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 complex matrix with different values
    matrix = np.array([[1 - 1j, 2 + 0j, 3 + 1j], [4 + 2j, 5 - 3j, 6 + 0j], [7 + 0j, 8 + 1j, 9 - 2j]], dtype=np.complex64)
    input_dict = {"matrix": tf.convert_to_tensor(matrix), "name": "adjoint_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2x2 float matrix (should still work as conjugate is a no-op)
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"matrix": tf.convert_to_tensor(matrix), "name": "adjoint_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D complex matrix
    matrix = np.array([[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], [[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]]], dtype=np.complex64)
    input_dict = {"matrix": tf.convert_to_tensor(matrix), "name": "adjoint_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2x2 complex matrix with negative values
    matrix = np.array([[-1 - 1j, -2 + 2j], [-3 + 3j, -4 - 4j]], dtype=np.complex64)
    input_dict = {"matrix": tf.convert_to_tensor(matrix), "name": "adjoint_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1x1 complex matrix
    matrix = np.array([[1 + 1j]], dtype=np.complex64)
    input_dict = {"matrix": tf.convert_to_tensor(matrix), "name": "adjoint_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4x4 complex matrix
    matrix = np.array([[1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j, 7 + 7j, 8 + 8j], [9 + 9j, 10 + 10j, 11 + 11j, 12 + 12j], [13 + 13j, 14 + 14j, 15 + 15j, 16 + 16j]], dtype=np.complex64)
    input_dict = {"matrix": tf.convert_to_tensor(matrix), "name": "adjoint_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128 matrix
    matrix = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex128)
    input_dict = {"matrix": tf.convert_to_tensor(matrix), "name": "adjoint_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 matrix
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"matrix": tf.convert_to_tensor(matrix), "name": "adjoint_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2x2 float16 matrix
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict = {"matrix": tf.convert_to_tensor(matrix), "name": "adjoint_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_inputs = tf_linalg_adjoint_inputs()
for i in range(len(temp_inputs)):
    temp_inputs[i]['matrix'] = temp_inputs[i]['matrix'].numpy()

generated_inputs["tf.linalg.adjoint"] = temp_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.adjoint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.adjoint'.")

check_valid('tf.linalg.adjoint', generated_inputs['tf.linalg.adjoint'], lib="tf", suffix=0)
