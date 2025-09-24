
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_matrix_diag_v3_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MatrixDiagV3 function.
    """
    list_of_inputs = []

    # Input 1: Main diagonal, inferred square matrix
    input_dict_1 = {
        'diagonal': np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32),
        'k': np.array(0, dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'align': 'RIGHT_LEFT',
        'name': 'main_diagonal_inferred'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Superdiagonal, inferred square matrix, float values
    input_dict_2 = {
        'diagonal': np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32),
        'k': np.array(1, dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(0.0, dtype=np.float32),
        'align': 'RIGHT_LEFT',
        'name': 'superdiagonal_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Subdiagonal, specified rectangular matrix, negative padding
    input_dict_3 = {
        'diagonal': np.array([10, 20, 30], dtype=np.int32),
        'k': np.array(-1, dtype=np.int32),
        'num_rows': np.array(4, dtype=np.int32),
        'num_cols': np.array(5, dtype=np.int32),
        'padding_value': np.array(-1, dtype=np.int32),
        'align': 'RIGHT_LEFT',
        'name': 'subdiagonal_rectangular'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Tridiagonal band, inferred shape, default alignment
    input_dict_4 = {
        'diagonal': np.array([[[0, 8, 9], [1, 2, 3], [4, 5, 0]], [[0, 2, 3], [6, 7, 9], [9, 1, 0]]], dtype=np.int32),
        'k': np.array([-1, 1], dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'align': 'RIGHT_LEFT',
        'name': 'tridiagonal_band_default_align'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tridiagonal band, inferred shape, LEFT_RIGHT alignment
    input_dict_5 = {
        'diagonal': np.array([[[8, 9, 0], [1, 2, 3], [0, 4, 5]], [[2, 3, 0], [6, 7, 9], [0, 9, 1]]], dtype=np.int32),
        'k': np.array([-1, 1], dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'align': 'LEFT_RIGHT',
        'name': 'tridiagonal_band_left_right_align'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Rectangular matrix, inferred num_cols
    input_dict_6 = {
        'diagonal': np.array([1, 2], dtype=np.int32),
        'k': np.array(-1, dtype=np.int32),
        'num_rows': np.array(3, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(9, dtype=np.int32),
        'align': 'RIGHT_LEFT',
        'name': 'rectangular_infer_cols'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Higher rank batch dimension, float64
    input_dict_7 = {
        'diagonal': np.arange(1, 7, dtype=np.float64).reshape(1, 2, 3),
        'k': np.array(0, dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(0.0, dtype=np.float64),
        'align': 'RIGHT_LEFT',
        'name': 'high_rank_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Superdiagonal band, LEFT_LEFT alignment
    input_dict_8 = {
        'diagonal': np.array([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]], dtype=np.int32),
        'k': np.array([0, 2], dtype=np.int32),
        'num_rows': np.array(4, dtype=np.int32),
        'num_cols': np.array(4, dtype=np.int32),
        'padding_value': np.array(-1, dtype=np.int32),
        'align': 'LEFT_LEFT',
        'name': 'superdiagonal_band_left_left'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Subdiagonal band, RIGHT_RIGHT alignment, rectangular
    input_dict_9 = {
        'diagonal': np.array([[0, 1, 2, 3], [0, 4, 5, 6], [7, 8, 9, 10]], dtype=np.int32),
        'k': np.array([-2, 0], dtype=np.int32),
        'num_rows': np.array(5, dtype=np.int32),
        'num_cols': np.array(4, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'align': 'RIGHT_RIGHT',
        'name': 'subdiagonal_band_right_right'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Single diagonal specified as a band k=(2,2)
    input_dict_10 = {
        'diagonal': np.array([[1, 2, 3]], dtype=np.int32),
        'k': np.array([2, 2], dtype=np.int32),
        'num_rows': np.array(5, dtype=np.int32),
        'num_cols': np.array(5, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'align': 'RIGHT_LEFT',
        'name': 'single_diag_as_band'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Wide band, large matrix
    input_dict_11 = {
        'diagonal': np.ones((2, 5, 8), dtype=np.int32),
        'k': np.array([-2, 2], dtype=np.int32),
        'num_rows': np.array(10, dtype=np.int32),
        'num_cols': np.array(8, dtype=np.int32),
        'padding_value': np.array(5, dtype=np.int32),
        'align': 'LEFT_RIGHT',
        'name': 'wide_band_large_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Zero-sized num_rows, should produce empty output
    input_dict_12 = {
        'diagonal': np.array([], dtype=np.float32),
        'k': np.array(0, dtype=np.int32),
        'num_rows': np.array(0, dtype=np.int32),
        'num_cols': np.array(5, dtype=np.int32),
        'padding_value': np.array(0.0, dtype=np.float32),
        'align': 'RIGHT_LEFT',
        'name': 'zero_rows'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixDiagV3"] = get_matrix_diag_v3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MatrixDiagV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiagV3'.")

check_valid('tf.raw_ops.MatrixDiagV3', generated_inputs['tf.raw_ops.MatrixDiagV3'], lib="tf", suffix=0)
