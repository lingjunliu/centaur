
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_LinearOperatorLowerTriangular_inputs():
    list_of_inputs = []

    input_dict1 = {
        'tril': np.array([[1., 2.], [3., 4.]]),
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'op1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        'tril': np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]),
        'is_non_singular': False,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'op2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        'tril': np.array([[-1., 0.], [0., -2.]]),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'op3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input_dict4 = {
        'tril': np.array([[[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]],
                          [[7., 0., 0.], [8., 9., 0.], [10., 11., 12.]]]),
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'op4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input_dict5 = {
        'tril': np.random.rand(3, 3),
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'op5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input_dict6 = {
        'tril': np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'op6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input_dict7 = {
        'tril': np.array([[[1., 2., 3.], [0., 4., 5.], [0., 0., 6.]]]),
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'op7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input_dict8 = {
        'tril': np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'op8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input_dict9 = {
        'tril': np.array([[[1., -2.], [-3., 4.]]]),
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'op9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input_dict10 = {
        'tril': np.array([[[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]]]),
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'op10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorLowerTriangular"] = tf_linalg_LinearOperatorLowerTriangular_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorLowerTriangular' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorLowerTriangular'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorLowerTriangular', generated_inputs['tf.linalg.LinearOperatorLowerTriangular'], lib="tf", suffix=0)
