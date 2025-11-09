
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_linalg_LinearOperatorZeros_inputs():
    list_of_inputs = []

    input_dict = {
        "num_rows": np.array(2, dtype=np.int32),
        "num_columns": np.array(2, dtype=np.int32),
        "batch_shape": np.array([], dtype=np.int32),
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_2x2_f32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(3, dtype=np.int64),
        "num_columns": np.array(5, dtype=np.int64),
        "batch_shape": np.array([4], dtype=np.int64),
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_3x5_batch4_f64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(0, dtype=np.int32),
        "num_columns": np.array(0, dtype=np.int32),
        "batch_shape": np.array([2, 3], dtype=np.int32),
        "dtype": np.float16,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "empty_0x0_b23_f16",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(0, dtype=np.int32),
        "num_columns": np.array(7, dtype=np.int32),
        "batch_shape": np.array([], dtype=np.int32),
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "zeros_0x7",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(9, dtype=np.int64),
        "num_columns": np.array(0, dtype=np.int64),
        "batch_shape": np.array([0], dtype=np.int64),
        "dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_9x0_c64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(1, dtype=np.int32),
        "num_columns": np.array(1, dtype=np.int32),
        "batch_shape": np.array([5, 1], dtype=np.int32),
        "dtype": np.complex128,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_1x1_b5x1_c128",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(10, dtype=np.int64),
        "num_columns": np.array(10, dtype=np.int64),
        "batch_shape": np.array([2, 0], dtype=np.int64),
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_10x10_b2x0_f32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(5, dtype=np.int32),
        "num_columns": np.array(3, dtype=np.int32),
        "batch_shape": np.array([3, 2], dtype=np.int32),
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_5x3_b3x2_f64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(128, dtype=np.int32),
        "num_columns": np.array(256, dtype=np.int32),
        "batch_shape": np.array([2], dtype=np.int32),
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "zeros_128x256_b2_f32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(4, dtype=np.int64),
        "num_columns": np.array(4, dtype=np.int64),
        "batch_shape": np.array([1, 1, 1], dtype=np.int64),
        "dtype": np.float16,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_4x4_b1x1x1_f16",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(2, dtype=np.int32),
        "num_columns": np.array(3, dtype=np.int32),
        "batch_shape": np.array([2, 1, 3], dtype=np.int32),
        "dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_2x3_b2x1x3_c64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(7, dtype=np.int64),
        "num_columns": np.array(7, dtype=np.int64),
        "batch_shape": np.array([], dtype=np.int64),
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_7x7_nobatch_f64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorZeros_2"] = tf_linalg_LinearOperatorZeros_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorZeros_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorZeros_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorZeros', generated_inputs['tf.linalg.LinearOperatorZeros_2'], lib="tf", suffix=2)
