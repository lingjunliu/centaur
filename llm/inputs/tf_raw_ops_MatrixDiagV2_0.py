
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixDiagV2_inputs():
    list_of_inputs = []

    # Input 1: Basic diagonal
    diagonal = np.array([1, 2, 3], dtype=np.int32)
    k = np.array(0, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(3, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Superdiagonal
    diagonal = np.array([1, 2], dtype=np.int32)
    k = np.array(1, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(3, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Subdiagonal
    diagonal = np.array([1, 2], dtype=np.int32)
    k = np.array(-1, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(3, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rectangular matrix
    diagonal = np.array([1, 2], dtype=np.int32)
    k = np.array(-1, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(4, dtype=np.int32)
    padding_value = np.array(9, dtype=np.int32)
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Band of diagonals
    diagonals = np.array([[1, 2, 3], [4, 5, 0]], dtype=np.int32)
    k = np.array([-1, 0], dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(3, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    input_dict = {"diagonal": diagonals, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched diagonal
    diagonal = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    k = np.array(0, dtype=np.int32)
    num_rows = np.array(4, dtype=np.int32)
    num_cols = np.array(4, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different padding value
    diagonal = np.array([1, 2, 3], dtype=np.float32)
    k = np.array(0, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(3, dtype=np.int32)
    padding_value = np.array(-1.0, dtype=np.float32)
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger k value
    diagonal = np.array([1], dtype=np.int32)
    k = np.array(2, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(3, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different datatype
    diagonal = np.array([1, 2, 3], dtype=np.int64)
    k = np.array(0, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(3, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int64)
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: k as range
    diagonal = np.array([[1, 2], [3, 4]], dtype=np.int32)
    k = np.array([0, 1], dtype=np.int32)
    num_rows = np.array(2, dtype=np.int32)
    num_cols = np.array(2, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixDiagV2"] = tf_raw_ops_MatrixDiagV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixDiagV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiagV2'.")

check_valid('tf.raw_ops.MatrixDiagV2', generated_inputs['tf.raw_ops.MatrixDiagV2'], lib="tf", suffix=0)
