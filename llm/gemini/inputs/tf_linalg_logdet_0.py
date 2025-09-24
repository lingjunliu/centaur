
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_logdet_inputs():
    list_of_inputs = []

    # Input 1: float32, simple 2x2
    matrix1 = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    input_dict1 = {"matrix": matrix1, "name": "logdet1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: float64, 3x3 with positive definite property
    matrix2 = np.array([[4.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 6.0]], dtype=np.float64)
    input_dict2 = {"matrix": matrix2, "name": "logdet2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: complex64, hermitian positive definite
    matrix3 = np.array([[4 + 0j, 1 - 1j], [1 + 1j, 5 + 0j]], dtype=np.complex64)
    input_dict3 = {"matrix": matrix3, "name": "logdet3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: float32, batch of matrices
    matrix5 = np.array([[[2.0, 0.0], [0.0, 3.0]], [[4.0, 0.0], [0.0, 5.0]]], dtype=np.float32)
    input_dict5 = {"matrix": matrix5, "name": "logdet5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 5: float64, a larger matrix
    matrix6 = np.random.rand(5, 5)
    matrix6 = np.dot(matrix6, matrix6.T)
    matrix6 = matrix6.astype(np.float64)
    input_dict6 = {"matrix": matrix6, "name": "logdet6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 6: complex128, hermitian matrix
    matrix7 = np.array([[3 + 0j, 2 - 1j], [2 + 1j, 4 + 0j]], dtype=np.complex128)
    input_dict7 = {"matrix": matrix7, "name": "logdet7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 7: float32, identity matrix
    matrix8 = np.eye(3, dtype=np.float32)
    input_dict8 = {"matrix": matrix8, "name": "logdet8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 8: float64, matrix with large values
    matrix9 = np.array([[1e6, 0.0], [0.0, 1e6]], dtype=np.float64)
    input_dict9 = {"matrix": matrix9, "name": "logdet9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 9: float32, 1x1 matrix
    matrix10 = np.array([[5.0]], dtype=np.float32)
    input_dict10 = {"matrix": matrix10, "name": "logdet10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 10: float64, positive definite
    matrix11 = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    input_dict11 = {"matrix": matrix11, "name": "logdet11"}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.logdet"] = tf_linalg_logdet_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.logdet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.logdet'.")

check_valid('tf.linalg.logdet', generated_inputs['tf.linalg.logdet'], lib="tf", suffix=0)
