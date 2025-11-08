
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_where_inputs():
    list_of_inputs = []

    # Input 1: bool 2D
    condition = tf.constant(np.array([[True, False], [True, False]], dtype=np.bool_))
    input_dict = {"name": "where_bool_2x2", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 3D
    condition = tf.constant(
        np.array(
            [
                [[1.5, 0.0], [-0.5, 0.0]],
                [[0.0, 0.25], [0.0, 0.75]],
                [[0.0, 0.0], [0.0, 0.01]],
            ],
            dtype=np.float32,
        )
    )
    input_dict = {"name": "where_float32_3d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 2D
    condition = tf.constant(
        np.array([[-1.0, 0.0, 3.14], [0.0, -2.71, 0.0]], dtype=np.float64)
    )
    input_dict = {"name": "where_float64_2d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32 empty 1D
    condition = tf.constant(np.array([], dtype=np.int32))
    input_dict = {"name": "where_int32_empty", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64 4D
    condition = tf.constant(
        np.array(
            [
                [[[0, 1], [0, 0]]],
                [[[2, 0], [0, 3]]],
            ],
            dtype=np.int64,
        )
    )
    input_dict = {"name": "where_int64_4d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8 1D
    condition = tf.constant(np.array([0, 255, 1, 0, 128], dtype=np.uint8))
    input_dict = {"name": "where_uint8_1d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int8 2D
    condition = tf.constant(
        np.array([[-1, 0, 1], [0, -128, 127]], dtype=np.int8)
    )
    input_dict = {"name": "where_int8_2d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64 2D
    condition = tf.constant(
        np.array([[1.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.5j, 0.0 + 0.0j]], dtype=np.complex64)
    )
    input_dict = {"name": "where_complex64_2d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128 3D
    condition = tf.constant(
        np.array(
            [
                [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 1.0j]],
                [[0.0 + 0.0j, 2.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
            ],
            dtype=np.complex128,
        )
    )
    input_dict = {"name": "where_complex128_3d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16 3D
    condition = tf.constant(
        np.array(
            [
                [[0, -5, 0], [10, 0, 0]],
                [[0, 0, 0], [0, 3, -2]],
            ],
            dtype=np.int16,
        )
    )
    input_dict = {"name": "where_int16_3d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: bool 3D all false
    condition = tf.constant(np.zeros((2, 3, 1), dtype=np.bool_))
    input_dict = {"name": "where_bool_3d_all_false", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: int32 5D small
    condition = tf.constant(
        np.array(
            [[[[[0, 1]]], [[[2, 0]]]]],
            dtype=np.int32,
        )
    )
    input_dict = {"name": "where_int32_5d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Where"] = tf_raw_ops_where_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Where' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Where'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Where', generated_inputs['tf.raw_ops.Where'], lib="tf", suffix=0)
