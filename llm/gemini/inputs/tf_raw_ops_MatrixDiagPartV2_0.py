
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixDiagPartV2_inputs():
    list_of_inputs = []

    # Input 1: float32 matrix with single diagonal (k = 0)
    list_of_inputs.append({
        "name": "diag_part_1",
        "input": np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=np.float32),
        "k": np.array(0, dtype=np.int32),
        "padding_value": np.array(0.0, dtype=np.float32)
    })

    # Input 2: int32 matrix with a single negative diagonal (k = -1)
    list_of_inputs.append({
        "name": "diag_part_2",
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32),
        "k": np.array(-1, dtype=np.int32),
        "padding_value": np.array(-1, dtype=np.int32)
    })

    # Input 3: float64 matrix with a band of diagonals (k = [-1, 1])
    list_of_inputs.append({
        "name": "diag_part_3",
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64),
        "k": np.array([-1, 1], dtype=np.int32),
        "padding_value": np.array(9.0, dtype=np.float64)
    })

    # Input 4: 3D float32 tensor with a superdiagonal band (k = [1, 3])
    list_of_inputs.append({
        "name": "diag_part_4",
        "input": np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]], [[5, 4, 3, 2], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32),
        "k": np.array([1, 3], dtype=np.int32),
        "padding_value": np.array(9.0, dtype=np.float32)
    })

    # Input 5: int64 matrix with a single negative diagonal (k = -1)
    list_of_inputs.append({
        "name": "diag_part_5",
        "input": np.arange(10, dtype=np.int64).reshape((2, 5)),
        "k": np.array(-1, dtype=np.int32),
        "padding_value": np.array(0, dtype=np.int64)
    })

    # Input 6: complex64 4D tensor with a diagonal band (k = [0, 1])
    list_of_inputs.append({
        "name": "diag_part_6",
        "input": np.arange(36, dtype=np.complex64).reshape((2, 2, 3, 3)),
        "k": np.array([0, 1], dtype=np.int32),
        "padding_value": np.array(0 + 0j, dtype=np.complex64)
    })

    # Input 7: uint8 matrix with a subdiagonal band (k = [-2, -1])
    list_of_inputs.append({
        "name": "diag_part_7",
        "input": np.ones((4, 4), dtype=np.uint8),
        "k": np.array([-2, -1], dtype=np.int32),
        "padding_value": np.array(255, dtype=np.uint8)
    })

    # Input 8: float32 3D tensor, main diagonal (k = 0)
    list_of_inputs.append({
        "name": "diag_part_8",
        "input": np.zeros((3, 1, 2), dtype=np.float32),
        "k": np.array(0, dtype=np.int32),
        "padding_value": np.array(-99.9, dtype=np.float32)
    })

    # Input 9: int32 matrix with a wider subdiagonal band (k = [-2, 0])
    list_of_inputs.append({
        "name": "diag_part_9",
        "input": np.arange(8, dtype=np.int32).reshape((4, 2)),
        "k": np.array([-2, 0], dtype=np.int32),
        "padding_value": np.array(999, dtype=np.int32)
    })

    # Input 10: int16 matrix with superdiagonals (k = [1, 2])
    list_of_inputs.append({
        "name": "diag_part_10",
        "input": np.arange(25, dtype=np.int16).reshape((5, 5)),
        "k": np.array([1, 2], dtype=np.int32),
        "padding_value": np.array(42, dtype=np.int16)
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixDiagPartV2"] = tf_raw_ops_MatrixDiagPartV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MatrixDiagPartV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiagPartV2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MatrixDiagPartV2', generated_inputs['tf.raw_ops.MatrixDiagPartV2'], lib="tf", suffix=0)
