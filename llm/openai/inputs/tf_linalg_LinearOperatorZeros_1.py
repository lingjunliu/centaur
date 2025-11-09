
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_LinearOperatorZeros_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "num_rows": 2,
        "num_columns": 2,
        "batch_shape": [],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_2x2_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "num_rows": 3,
        "num_columns": 4,
        "batch_shape": [],
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "zeros_3x4_f64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "num_rows": 2,
        "num_columns": 2,
        "batch_shape": [2],
        "dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_batch2_2x2_c64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "num_rows": 5,
        "num_columns": 5,
        "batch_shape": [2, 3],
        "dtype": np.complex128,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_b23_5x5_c128"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "num_rows": 7,
        "num_columns": 7,
        "batch_shape": [0],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_emptybatch_7x7_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "num_rows": 1,
        "num_columns": 3,
        "batch_shape": [4],
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "zeros_b4_1x3_f64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "num_rows": 0,
        "num_columns": 0,
        "batch_shape": [],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_0x0_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "num_rows": 2,
        "num_columns": 2,
        "batch_shape": [1, 0, 2],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_b102_2x2_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "num_rows": 1000,
        "num_columns": 1000,
        "batch_shape": [],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_1000x1000_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "num_rows": 8,
        "num_columns": 2,
        "batch_shape": [5, 1],
        "dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_b51_8x2_c64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        "num_rows": 4,
        "num_columns": 4,
        "batch_shape": [3],
        "dtype": np.float16,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_b3_4x4_f16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_dict = {
        "num_rows": 6,
        "num_columns": 6,
        "batch_shape": [1, 1, 1],
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_b111_6x6_f64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorZeros_1"] = tf_linalg_LinearOperatorZeros_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorZeros_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorZeros_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorZeros', generated_inputs['tf.linalg.LinearOperatorZeros_1'], lib="tf", suffix=1)
