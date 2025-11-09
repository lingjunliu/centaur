
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorzeros_inputs():
    list_of_inputs = []

    input_dict = {
        "num_rows": 2,
        "num_columns": 2,
        "batch_shape": (),
        "dtype": np.dtype('float32'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_2x2_f32_no_batch"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 3,
        "num_columns": 5,
        "batch_shape": (4,),
        "dtype": np.dtype('float64'),
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_3x5_f64_b4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 0,
        "num_columns": 0,
        "batch_shape": (),
        "dtype": np.dtype('float32'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_0x0_f32_empty"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 1,
        "num_columns": 1,
        "batch_shape": (2,),
        "dtype": np.dtype('complex64'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_1x1_c64_b2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 7,
        "num_columns": 3,
        "batch_shape": (2, 3),
        "dtype": np.dtype('float16'),
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_7x3_f16_b2x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 10,
        "num_columns": 10,
        "batch_shape": (1,),
        "dtype": np.dtype('complex128'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_10x10_c128_b1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 2,
        "num_columns": 0,
        "batch_shape": (3,),
        "dtype": np.dtype('float32'),
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "zeros_2x0_f32_b3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 0,
        "num_columns": 5,
        "batch_shape": (0,),
        "dtype": np.dtype('float64'),
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_0x5_f64_b0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 4,
        "num_columns": 4,
        "batch_shape": (3, 1, 2),
        "dtype": np.dtype('float64'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_4x4_f64_b3x1x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 5,
        "num_columns": 5,
        "batch_shape": (),
        "dtype": np.dtype('complex64'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_5x5_c64_assert"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 6,
        "num_columns": 6,
        "batch_shape": (2, 2),
        "dtype": np.dtype('float32'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_6x6_f32_b2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorZeros_3"] = tf_linalg_linearoperatorzeros_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorZeros_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorZeros_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorZeros', generated_inputs['tf.linalg.LinearOperatorZeros_3'], lib="tf", suffix=3)
