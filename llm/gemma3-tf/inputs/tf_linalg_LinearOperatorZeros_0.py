
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def linear_operator_zeros_inputs():
    list_of_inputs = []

    input_dict = {
        'num_rows': 2,
        'num_columns': 3,
        'batch_shape': [],
        'dtype': np.float32,
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'assert_proper_shapes': False,
        'name': 'LinearOperatorZeros_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'num_rows': 5,
        'num_columns': 5,
        'batch_shape': [2, 3],
        'dtype': np.complex64,
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'assert_proper_shapes': True,
        'name': 'LinearOperatorZeros_2'
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
