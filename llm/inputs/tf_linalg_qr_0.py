
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_qr_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 matrix, full_matrices=False
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    full_matrices = False
    name = "qr_decomposition_1"
    input_dict = {"input": input_tensor, "full_matrices": full_matrices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 matrix, full_matrices=True
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    full_matrices = True
    name = "qr_decomposition_2"
    input_dict = {"input": input_tensor, "full_matrices": full_matrices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half matrix, full_matrices=False
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float16)
    full_matrices = False
    name = "qr_decomposition_3"
    input_dict = {"input": input_tensor, "full_matrices": full_matrices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64 matrix, full_matrices=True
    input_tensor = np.array([[1.0+1j, 2.0+2j], [3.0+3j, 4.0+4j]], dtype=np.complex64)
    full_matrices = True
    name = "qr_decomposition_4"
    input_dict = {"input": input_tensor, "full_matrices": full_matrices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128 matrix, full_matrices=False
    input_tensor = np.array([[1.0+1j, 2.0+2j, 3.0+3j], [4.0+4j, 5.0+5j, 6.0+6j]], dtype=np.complex128)
    full_matrices = False
    name = "qr_decomposition_5"
    input_dict = {"input": input_tensor, "full_matrices": full_matrices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 tensor, full_matrices=True
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    full_matrices = True
    name = "qr_decomposition_6"
    input_dict = {"input": input_tensor, "full_matrices": full_matrices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 matrix with negative values, full_matrices=False
    input_tensor = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    full_matrices = False
    name = "qr_decomposition_7"
    input_dict = {"input": input_tensor, "full_matrices": full_matrices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 matrix, rectangular, full_matrices=True
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    full_matrices = True
    name = "qr_decomposition_8"
    input_dict = {"input": input_tensor, "full_matrices": full_matrices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 matrix, singular matrix, full_matrices=False.  The gradient is not well defined here.
    input_tensor = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    full_matrices = False
    name = "qr_decomposition_9"
    input_dict = {"input": input_tensor, "full_matrices": full_matrices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  4D float32 tensor, full_matrices=True
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    full_matrices = True
    name = "qr_decomposition_10"
    input_dict = {"input": input_tensor, "full_matrices": full_matrices, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.qr"] = tf_linalg_qr_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.qr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.qr'.")

check_valid('tf.linalg.qr', generated_inputs['tf.linalg.qr'], lib="tf", suffix=0)
