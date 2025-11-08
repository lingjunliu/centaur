
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def linear_operator_zeros_inputs():
    list_of_inputs = []
    
    # Input 1: Basic zero matrix with num_rows=2
    input_dict = {
        "num_rows": 2,
        "num_column": None,
        "batch_shape": None,
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Square zero matrix with num_rows=3
    input_dict = {
        "num_rows": 3,
        "num_column": None,
        "batch_shape": None,
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Zero matrix with num_rows=4, num_columns=5
    input_dict = {
        "num_rows": 4,
        "num_column": 5,
        "batch_shape": None,
        "dtype": np.int32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Batched zero matrix with batch_shape=[2]
    input_dict = {
        "num_rows": 2,
        "num_column": None,
        "batch_shape": [2],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Batched zero matrix with batch_shape=[3, 2]
    input_dict = {
        "num_rows": 2,
        "num_column": None,
        "batch_shape": [3, 2],
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Zero matrix with negative num_rows (invalid but accepted)
    input_dict = {
        "num_rows": -1,
        "num_column": None,
        "batch_shape": None,
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Zero matrix with num_rows=0 (edge case)
    input_dict = {
        "num_rows": 0,
        "num_column": None,
        "batch_shape": None,
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Zero matrix with num_rows=5, num_columns=6
    input_dict = {
        "num_rows": 5,
        "num_column": 6,
        "batch_shape": None,
        "dtype": np.int64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Zero matrix with batch_shape=[1]
    input_dict = {
        "num_rows": 2,
        "num_column": None,
        "batch_shape": [1],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Zero matrix with num_rows=3, batch_shape=[2, 2]
    input_dict = {
        "num_rows": 3,
        "num_column": None,
        "batch_shape": [2, 2],
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "LinearOperatorZeros"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorZeros"] = linear_operator_zeros_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorZeros' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorZeros'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorZeros', generated_inputs['tf.linalg.LinearOperatorZeros'], lib="tf", suffix=0)
