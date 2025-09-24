
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_matrix_diag_v2_inputs():
    list_of_inputs = []

    # Input 1: Main diagonal, inferred square matrix
    diagonal = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array(0, dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'name': 'main_diagonal_inferred_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Superdiagonal, inferred square matrix
    diagonal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array(1, dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'name': 'superdiagonal_inferred_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Band of diagonals, inferred square matrix
    diagonal = np.array([[[1, 2, 3], [4, 5, 0]], [[6, 7, 9], [9, 1, 0]]], dtype=np.int32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array([-1, 0], dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'name': 'band_of_diagonals_inferred_square'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Subdiagonal, specified rectangular matrix
    diagonal = np.array([1, 2], dtype=np.int32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array(-1, dtype=np.int32),
        'num_rows': np.array(3, dtype=np.int32),
        'num_cols': np.array(4, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'name': 'subdiagonal_specified_rectangular'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Subdiagonal, inferred columns, custom padding
    diagonal = np.array([1, 2], dtype=np.int32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array(-1, dtype=np.int32),
        'num_rows': np.array(3, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(9, dtype=np.int32),
        'name': 'subdiagonal_inferred_cols_custom_padding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float32 diagonal with negative padding
    diagonal = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array(0, dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(-1.0, dtype=np.float32),
        'name': 'float32_diagonal_negative_padding'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: High-rank batch with a single diagonal
    diagonal = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array(0, dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'name': 'high_rank_batch_single_diagonal'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High-rank batch with a band of diagonals (float64) - FIXED
    diagonal = np.random.rand(1, 2, 3, 5).astype(np.float64)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array([-1, 1], dtype=np.int32),
        'num_rows': np.array(5, dtype=np.int32),
        'num_cols': np.array(5, dtype=np.int32),
        'padding_value': np.array(-99.9, dtype=np.float64),
        'name': 'high_rank_batch_band_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Wide rectangular matrix - FIXED
    diagonal = np.array([10, 20, 30, 40], dtype=np.int32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array(2, dtype=np.int32),
        'num_rows': np.array(4, dtype=np.int32),
        'num_cols': np.array(6, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'name': 'wide_rectangular_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tall rectangular matrix with negative k - FIXED
    diagonal = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array(-2, dtype=np.int32),
        'num_rows': np.array(7, dtype=np.int32),
        'num_cols': np.array(5, dtype=np.int32),
        'padding_value': np.array(-1, dtype=np.int32),
        'name': 'tall_rectangular_matrix_negative_k'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Minimal 1x1 case
    diagonal = np.array([5], dtype=np.int32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array(0, dtype=np.int32),
        'num_rows': np.array(1, dtype=np.int32),
        'num_cols': np.array(1, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'name': 'minimal_1x1_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Single diagonal specified as a band (k[0] == k[1])
    diagonal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {
        'diagonal': diagonal,
        'k': np.array([1, 1], dtype=np.int32),
        'num_rows': np.array(-1, dtype=np.int32),
        'num_cols': np.array(-1, dtype=np.int32),
        'padding_value': np.array(0, dtype=np.int32),
        'name': 'single_diagonal_as_band'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixDiagV2"] = get_tf_raw_ops_matrix_diag_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MatrixDiagV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiagV2'.")

check_valid('tf.raw_ops.MatrixDiagV2', generated_inputs['tf.raw_ops.MatrixDiagV2'], lib="tf", suffix=0)
