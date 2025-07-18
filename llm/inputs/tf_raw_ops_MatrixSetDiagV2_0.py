
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import torch # As per instruction, though numpy is used to create tensors


def get_tf_raw_ops_MatrixSetDiagV2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MatrixSetDiagV2 function.
    """
    list_of_inputs = []

    # Input 1: Main diagonal (k=0) for a 3D input tensor.
    input_1 = np.arange(1, 25, dtype=np.int32).reshape(2, 3, 4)
    diagonal_1 = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    k_1 = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "main_diagonal_3d",
            "input": input_1,
            "diagonal": diagonal_1,
            "k": k_1,
        }
    )

    # Input 2: Superdiagonal (k=1) for a 3D float tensor.
    input_2 = np.ones((2, 3, 4), dtype=np.float32)
    diagonal_2 = np.array(
        [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32
    )
    k_2 = np.array(1, dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "superdiagonal_3d_float",
            "input": input_2,
            "diagonal": diagonal_2,
            "k": k_2,
        }
    )

    # Input 3: Subdiagonal (k=-2) for a 3D tensor.
    input_3 = np.zeros((1, 5, 4), dtype=np.int32)
    diagonal_3 = np.array([[99, 88, 77]], dtype=np.int32)
    k_3 = np.array(-2, dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "subdiagonal_3d",
            "input": input_3,
            "diagonal": diagonal_3,
            "k": k_3,
        }
    )

    # Input 4: Band of diagonals k=(-1, 1) for a 3D tensor.
    input_4 = np.full((1, 4, 5), 7, dtype=np.int32)
    diagonal_4 = np.array(
        [[[1, 2, 3, 0], [5, 6, 7, 8], [9, 8, 7, 6]]], dtype=np.int32
    )
    k_4 = np.array([-1, 1], dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "band_3d",
            "input": input_4,
            "diagonal": diagonal_4,
            "k": k_4
        }
    )

    # Input 5: k is a pair where k[0] == k[1], should act as scalar k.
    input_5 = np.zeros((1, 5, 5), dtype=np.float32)
    diagonal_5 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    k_5 = np.array([2, 2], dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "pair_k_equal",
            "input": input_5,
            "diagonal": diagonal_5,
            "k": k_5
        }
    )

    # Input 6: Higher rank input (4D) with a band of diagonals.
    input_6 = np.ones((1, 2, 3, 4), dtype=np.int32)
    diagonal_6 = np.array(
        [[[[1, 2, 3], [4, 5, 0]], [[6, 7, 8], [9, 1, 0]]]],
        dtype=np.int32,
    )
    k_6 = np.array([-1, 0], dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "4d_input_band",
            "input": input_6,
            "diagonal": diagonal_6,
            "k": k_6
        }
    )

    # Input 7: Minimum rank input (2D) with main diagonal.
    input_7 = np.arange(20, dtype=np.int32).reshape(4, 5)
    diagonal_7 = np.array([100, 200, 300, 400], dtype=np.int32)
    k_7 = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "2d_input_main_diag",
            "input": input_7,
            "diagonal": diagonal_7,
            "k": k_7,
        }
    )

    # Input 8: Minimum rank input (2D) with a band of diagonals.
    input_8 = np.zeros((4, 5), dtype=np.float32)
    diagonal_8 = np.array(
        [[1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 0.0]], dtype=np.float32
    )
    k_8 = np.array([-1, 0], dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "2d_input_band",
            "input": input_8,
            "diagonal": diagonal_8,
            "k": k_8
        }
    )

    # Input 9: Tall matrix (M > N) with a subdiagonal band.
    input_9 = np.full((1, 5, 3), 9, dtype=np.int32)
    diagonal_9 = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32)
    k_9 = np.array([-2, -1], dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "tall_matrix_subdiagonals",
            "input": input_9,
            "diagonal": diagonal_9,
            "k": k_9,
        }
    )

    # Input 10: Wide matrix (N > M) with a superdiagonal band.
    input_10 = np.full((2, 3, 6), 8, dtype=np.int32)
    diagonal_10 = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]]],
        dtype=np.int32,
    )
    k_10 = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "wide_matrix_superdiagonals",
            "input": input_10,
            "diagonal": diagonal_10,
            "k": k_10,
        }
    )
    
    # Input 11: 2D input with k=[0, 0] band.
    input_11 = np.zeros((3, 3), dtype=np.int32)
    diagonal_11 = np.array([1, 2, 3], dtype=np.int32)
    k_11 = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(
        {
            "name": "k_zero_band_2d",
            "input": input_11,
            "diagonal": diagonal_11,
            "k": k_11,
        }
    )

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixSetDiagV2"] = get_tf_raw_ops_MatrixSetDiagV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MatrixSetDiagV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixSetDiagV2'.")

check_valid('tf.raw_ops.MatrixSetDiagV2', generated_inputs['tf.raw_ops.MatrixSetDiagV2'], lib="tf", suffix=0)
