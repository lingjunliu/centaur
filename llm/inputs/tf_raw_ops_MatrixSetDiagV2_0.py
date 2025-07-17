
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_matrix_set_diag_v2_inputs():
    list_of_inputs = []

    # Input 1: Basic case, main diagonal
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.int32)
    diagonal_tensor = np.array([[10, 11, 12]]).astype(np.int32)
    k_tensor = np.array(0).astype(np.int32)

    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "k": k_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Superdiagonal (k=1)
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.int32)
    diagonal_tensor = np.array([[2, 3]]).astype(np.int32)
    k_tensor = np.array(1).astype(np.int32)

    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "k": k_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Subdiagonal (k=-1)
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.int32)
    diagonal_tensor = np.array([[4, 5]]).astype(np.int32)
    k_tensor = np.array(-1).astype(np.int32)

    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "k": k_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch input, main diagonal
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]]]).astype(np.int32)
    diagonal_tensor = np.array([[10, 11, 12], [20, 21, 22]]).astype(np.int32)
    k_tensor = np.array(0).astype(np.int32)

    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "k": k_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: k is a tuple (banded diagonal), k=(-1, 0)
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.int32)
    diagonal_tensor = np.array([[[4, 5], [7, 8]]]).astype(np.int32)
    k_tensor = np.array([-1, 0]).astype(np.int32)

    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "k": k_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: k is a tuple (banded diagonal), k=(0, 1)
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.int32)
    diagonal_tensor = np.array([[[2, 3], [5, 6]]]).astype(np.int32)
    k_tensor = np.array([0, 1]).astype(np.int32)

    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "k": k_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Larger input and diagonal sizes, main diagonal
    input_tensor = np.random.randint(0, 10, size=(2, 5, 5)).astype(np.int32)
    diagonal_tensor = np.random.randint(0, 10, size=(2, 5)).astype(np.int32)
    k_tensor = np.array(0).astype(np.int32)

    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "k": k_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data type (float32)
    input_tensor = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(np.float32)
    diagonal_tensor = np.array([[10.0, 11.0, 12.0]]).astype(np.float32)
    k_tensor = np.array(0).astype(np.int32)

    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "k": k_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D Input
    input_tensor = np.random.randint(0, 10, size=(2, 2, 3, 3)).astype(np.int32)
    diagonal_tensor = np.random.randint(0, 10, size=(2, 2, 3)).astype(np.int32)
    k_tensor = np.array(0).astype(np.int32)

    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "k": k_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: k is a tuple with negative and positive values
    input_tensor = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]).astype(np.int32)
    diagonal_tensor = np.array([[[2,3], [6,7], [10,11]]]).astype(np.int32)
    k_tensor = np.array([-1, 1]).astype(np.int32)

    input_dict = {"input": input_tensor, "diagonal": diagonal_tensor, "k": k_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixSetDiagV2"] = tf_raw_ops_matrix_set_diag_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixSetDiagV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixSetDiagV2'.")

check_valid('tf.raw_ops.MatrixSetDiagV2', generated_inputs['tf.raw_ops.MatrixSetDiagV2'], lib="tf", suffix=0)
