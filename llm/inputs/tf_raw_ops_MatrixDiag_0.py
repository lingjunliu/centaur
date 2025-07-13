
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_matrix_diag_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D diagonal
    diagonal = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D diagonal
    diagonal = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D diagonal
    diagonal = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (float64)
    diagonal = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data type (complex64)
    diagonal = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D diagonal with negative values
    diagonal = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D diagonal with mixed positive and negative values
    diagonal = np.array([[-1, 2], [3, -4]], dtype=np.float32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty diagonal (valid, but will produce a zero tensor)
    diagonal = np.array([], dtype=np.int32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Specify a name
    diagonal = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": "my_matrix_diag"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher dimension with different shape
    diagonal = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    input_dict = {"diagonal": tf.convert_to_tensor(diagonal), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixDiag"] = tf_raw_ops_matrix_diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiag'.")

check_valid('tf.raw_ops.MatrixDiag', generated_inputs['tf.raw_ops.MatrixDiag'], lib="tf", suffix=0)
