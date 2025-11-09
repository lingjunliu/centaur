
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_MatrixSetDiag_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    diagonal_arr = np.array([9, 8], dtype=np.int32)
    input_dict = {"name": "msd_int32_square_2x2", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.arange(15, dtype=np.float32).reshape(3, 5)
    diagonal_arr = np.array([10.5, -2.0, 3.3], dtype=np.float32)
    input_dict = {"name": "msd_float32_rect_3x5", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = np.arange(18, dtype=np.int64).reshape(2, 3, 3)
    diagonal_arr = np.array([[1, -1, 1], [2, -2, 2]], dtype=np.int64)
    input_dict = {"name": "msd_int64_batch_2_3x3", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = (np.arange(2 * 1 * 3 * 2, dtype=np.float16).reshape(2, 1, 3, 2) / 10.0).astype(np.float16)
    diagonal_arr = np.array([[[0.5, -1.5]], [[2.0, 3.0]]], dtype=np.float16)
    input_dict = {"name": "msd_float16_batch_2_1_3x2", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = (np.arange(16, dtype=np.float32).reshape(4, 4) + 1j * np.arange(16, dtype=np.float32).reshape(4, 4)).astype(np.complex64)
    diagonal_arr = np.array([1 + 1j, -2 + 0.5j, 3 - 3j, -4 + 2j], dtype=np.complex64)
    input_dict = {"name": "msd_complex64_4x4", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = (np.arange(3 * 2 * 4, dtype=np.float64).reshape(3, 2, 4) - 5.0).astype(np.float64)
    diagonal_arr = np.array([[1.1, -2.2], [3.3, -4.4], [5.5, -6.6]], dtype=np.float64)
    input_dict = {"name": "msd_float64_batch_3_2x4", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = (np.arange(35, dtype=np.int32).reshape(5, 7) * -1).astype(np.int32)
    diagonal_arr = np.array([-9, -8, -7, -6, -5], dtype=np.int32)
    input_dict = {"name": "msd_int32_rect_5x7", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = (np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4) / 3.0).astype(np.float32)
    diagonal_arr = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"name": "msd_float32_large_batch_2_3_4x4", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = np.zeros((3, 3), dtype=np.float32)
    diagonal_arr = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    input_dict = {"name": "msd_float32_with_nans_infs", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = (np.arange(2 * 2 * 2, dtype=np.float64).reshape(2, 2, 2) + 1j * (np.arange(2 * 2 * 2, dtype=np.float64).reshape(2, 2, 2) + 1)).astype(np.complex128)
    diagonal_arr = np.array([[1 + 0j, 2 - 1j], [-1 + 2j, 0 + 0j]], dtype=np.complex128)
    input_dict = {"name": "msd_complex128_batch_2_2x2", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = np.array([[3.14, 2.71, -1.0, 0.0]], dtype=np.float32)
    diagonal_arr = np.array([42.0], dtype=np.float32)
    input_dict = {"name": "msd_float32_min_diag_len1_row", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    diagonal_arr = np.array([7.0], dtype=np.float64)
    input_dict = {"name": "msd_float64_min_diag_len1_col", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixSetDiag"] = tf_raw_ops_MatrixSetDiag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MatrixSetDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixSetDiag'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MatrixSetDiag', generated_inputs['tf.raw_ops.MatrixSetDiag'], lib="tf", suffix=0)
