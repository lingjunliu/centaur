
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_matrix_set_diag_inputs():
    list_of_inputs = []

    # Case 1: 2D input, float32, square matrix
    input_1 = np.zeros((3, 3), dtype=np.float32)
    diagonal_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({
        'name': 'float32_square_2d',
        'input': input_1,
        'diagonal': diagonal_1
    })

    # Case 2: 2D input, int32, rectangular M > N
    input_2 = np.ones((4, 3), dtype=np.int32)
    diagonal_2 = np.array([-1, -2, -3], dtype=np.int32)
    list_of_inputs.append({
        'name': 'int32_rect_m_greater_n',
        'input': input_2,
        'diagonal': diagonal_2
    })

    # Case 3: 2D input, float64, rectangular M < N with negative values
    input_3 = np.array([[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]], dtype=np.float64)
    diagonal_3 = np.array([10.0, 20.0], dtype=np.float64)
    list_of_inputs.append({
        'name': 'float64_rect_m_less_n',
        'input': input_3,
        'diagonal': diagonal_3
    })

    # Case 4: 3D input, float32, batch of square matrices
    input_4 = np.random.randn(2, 3, 3).astype(np.float32)
    diagonal_4 = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append({
        'name': 'float32_batch_3d',
        'input': input_4,
        'diagonal': diagonal_4
    })

    # Case 5: 3D input, int64, batch of rectangular matrices
    input_5 = np.zeros((2, 2, 4), dtype=np.int64)
    diagonal_5 = np.ones((2, 2), dtype=np.int64)
    list_of_inputs.append({
        'name': 'int64_batch_3d',
        'input': input_5,
        'diagonal': diagonal_5
    })

    # Case 6: 4D input, complex64
    input_6 = np.ones((2, 2, 3, 3), dtype=np.complex64)
    diagonal_6 = np.zeros((2, 2, 3), dtype=np.complex64)
    list_of_inputs.append({
        'name': 'complex64_4d',
        'input': input_6,
        'diagonal': diagonal_6
    })

    # Case 7: 2D input, bool
    input_7 = np.array([[True, False], [False, True]], dtype=bool)
    diagonal_7 = np.array([False, False], dtype=bool)
    list_of_inputs.append({
        'name': 'bool_2d',
        'input': input_7,
        'diagonal': diagonal_7
    })

    # Case 8: 3D input, float32, larger dimensions
    input_8 = np.full((1, 5, 5), -1.0, dtype=np.float32)
    diagonal_8 = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=np.float32)
    list_of_inputs.append({
        'name': 'float32_large_3d',
        'input': input_8,
        'diagonal': diagonal_8
    })

    # Case 9: 2D input, uint8, small rectangular
    input_9 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    diagonal_9 = np.array([10, 20], dtype=np.uint8)
    list_of_inputs.append({
        'name': 'uint8_rect_2d',
        'input': input_9,
        'diagonal': diagonal_9
    })

    # Case 10: 4D input, int32, 1x1 batch size
    input_10 = np.ones((1, 1, 2, 2), dtype=np.int32)
    diagonal_10 = np.array([[[9, 9]]], dtype=np.int32)
    list_of_inputs.append({
        'name': 'int32_1x1_batch_4d',
        'input': input_10,
        'diagonal': diagonal_10
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixSetDiag"] = tf_matrix_set_diag_inputs()

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
