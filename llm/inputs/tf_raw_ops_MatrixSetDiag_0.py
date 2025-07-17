
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_matrix_set_diag_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    diagonal_tensor = np.array([5, 6], dtype=np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "basic"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batched input
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    diagonal_tensor = np.array([[9, 10], [11, 12]], dtype=np.float32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "batched"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data type (float64)
    input_tensor = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    diagonal_tensor = np.array([5.5, 6.6], dtype=np.float64)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (int64)
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    diagonal_tensor = np.array([5, 6], dtype=np.int64)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D input
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.int32)
    diagonal_tensor = np.array([[10, 11, 12]], dtype=np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger matrix
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    diagonal_tensor = np.array([10, 11, 12], dtype=np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "larger"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batched, different shape matrices
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    diagonal_tensor = np.array([[9, 10], [11, 12]], dtype=np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "batched2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single batch
    input_tensor = np.array([[[1, 2], [3, 4]]], dtype=np.int32)
    diagonal_tensor = np.array([[5, 6]], dtype=np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "singlebatch"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rectangular input matrix
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    diagonal_tensor = np.array([7, 8], dtype=np.int32)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "rectangular"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex data type
    input_tensor = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    diagonal_tensor = np.array([5+5j, 6+6j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "name": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixSetDiag"] = tf_raw_ops_matrix_set_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixSetDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixSetDiag'.")

check_valid('tf.raw_ops.MatrixSetDiag', generated_inputs['tf.raw_ops.MatrixSetDiag'], lib="tf", suffix=0)
