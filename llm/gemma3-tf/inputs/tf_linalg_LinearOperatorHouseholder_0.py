
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_LinearOperatorHouseholder_inputs():
    list_of_inputs = []

    input_dict1 = {
        'reflection_axis': np.array([1.0, 1.0], dtype=np.float32),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'householder_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        'reflection_axis': np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'householder_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        'reflection_axis': np.array([0.5, -0.5, 0.7, -0.7], dtype=np.float32),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'householder_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input_dict4 = {
        'reflection_axis': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'householder_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorHouseholder"] = tf_linalg_LinearOperatorHouseholder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorHouseholder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorHouseholder'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorHouseholder', generated_inputs['tf.linalg.LinearOperatorHouseholder'], lib="tf", suffix=0)
