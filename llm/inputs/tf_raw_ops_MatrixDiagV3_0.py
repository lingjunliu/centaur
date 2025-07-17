
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixDiagV3_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix diagonal
    diagonal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    k = np.array(0, dtype=np.int32)
    num_rows = np.array(0, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Superdiagonal
    diagonal = np.array([[1, 2], [3, 4]], dtype=np.int32)
    k = np.array(1, dtype=np.int32)
    num_rows = np.array(0, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Subdiagonal
    diagonal = np.array([[1, 2], [3, 4]], dtype=np.int32)
    k = np.array(-1, dtype=np.int32)
    num_rows = np.array(0, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: k as a tuple
    diagonal = np.array([[[0, 8, 9], [1, 2, 3], [4, 5, 0]], [[0, 2, 3], [6, 7, 9], [9, 1, 0]]], dtype=np.int32)
    k = np.array([-1, 1], dtype=np.int32)
    num_rows = np.array(0, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different alignment
    diagonal = np.array([[[8, 9, 0], [1, 2, 3], [0, 4, 5]], [[2, 3, 0], [6, 7, 9], [0, 9, 1]]], dtype=np.int32)
    k = np.array([-1, 1], dtype=np.int32)
    num_rows = np.array(0, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "LEFT_RIGHT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rectangular matrix
    diagonal = np.array([1, 2], dtype=np.int32)
    k = np.array(-1, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(4, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rectangular matrix with padding_value
    diagonal = np.array([1, 2], dtype=np.int32)
    k = np.array(-1, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(9, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float type diagonal
    diagonal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    k = np.array(0, dtype=np.int32)
    num_rows = np.array(0, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(0.0, dtype=np.float32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative padding value
    diagonal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    k = np.array(0, dtype=np.int32)
    num_rows = np.array(0, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(-1, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shape for diagonal
    diagonal = np.array([1, 2, 3], dtype=np.int32)
    k = np.array(0, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Different shape for diagonal and k
    diagonal = np.array([1, 2], dtype=np.int32)
    k = np.array(1, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Example from documentation
    diagonal = np.array([1, 2], dtype=np.int32)
    k = np.array(-1, dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(4, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: Another example to test num_rows and num_cols inference
    diagonal = np.array([1, 2, 3, 4], dtype=np.int32)
    k = np.array(0, dtype=np.int32)
    num_rows = np.array(0, dtype=np.int32)
    num_cols = np.array(0, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: k as a tuple with num_rows/cols
    diagonal = np.array([[[0, 8, 9], [1, 2, 3], [4, 5, 0]], [[0, 2, 3], [6, 7, 9], [9, 1, 0]]], dtype=np.int32)
    k = np.array([-1, 1], dtype=np.int32)
    num_rows = np.array(3, dtype=np.int32)
    num_cols = np.array(3, dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    align = "RIGHT_LEFT"
    input_dict = {"diagonal": diagonal, "k": k, "num_rows": num_rows, "num_cols": num_cols, "padding_value": padding_value, "align": align, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatrixDiagV3"] = tf_raw_ops_MatrixDiagV3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatrixDiagV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiagV3'.")

check_valid('tf.raw_ops.MatrixDiagV3', generated_inputs['tf.raw_ops.MatrixDiagV3'], lib="tf", suffix=0)
