
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linear_operator_tridiag_inputs():
    list_of_inputs = []

    # Input 1
    diagonals = np.array([[1., 2.], [3., 4.], [5., 6.]])
    input_dict = {
        "diagonals": tf.constant(diagonals),
        "diagonals_format": "compact",
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "tridiag_op_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    diagonals = np.array([[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]])
    input_dict = {
        "diagonals": tf.constant(diagonals),
        "diagonals_format": "compact",
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "tridiag_op_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    diagonals = np.array([[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]], dtype=np.float64)
    input_dict = {
        "diagonals": tf.constant(diagonals),
        "diagonals_format": "compact",
        "is_non_singular": None,
        "is_self_adjoint": None,
        "is_positive_definite": None,
        "is_square": None,
        "name": "tridiag_op_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    diagonals = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    input_dict = {
        "diagonals": tf.constant(diagonals),
        "diagonals_format": "compact",
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "tridiag_op_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    diagonals = np.array([[-1., -2.], [-3., -4.], [-5., -6.]])
    input_dict = {
        "diagonals": tf.constant(diagonals),
        "diagonals_format": "compact",
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "tridiag_op_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    diagonals = np.array([[[1., 0.], [0., 1.], [1., 0.]], [[0., 1.], [1., 0.], [0., 1.]]])
    input_dict = {
        "diagonals": tf.constant(diagonals),
        "diagonals_format": "compact",
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "tridiag_op_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    diagonals = np.array([[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]], dtype=np.complex64)
    input_dict = {
        "diagonals": tf.constant(diagonals),
        "diagonals_format": "compact",
        "is_non_singular": None,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "tridiag_op_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    diagonals = np.array([[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]])
    input_dict = {
        "diagonals": tf.constant(diagonals),
        "diagonals_format": "compact",
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": True,
        "is_square": True,
        "name": "tridiag_op_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    diagonals = np.array([[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]])
    input_dict = {
        "diagonals": tf.constant(diagonals),
        "diagonals_format": "compact",
        "is_non_singular": None,
        "is_self_adjoint": True,
        "is_positive_definite": None,
        "is_square": True,
        "name": "tridiag_op_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    diagonals = np.array([[[1., 2.], [3., 4.], [5., 6.]]])
    input_dict = {
        "diagonals": tf.constant(diagonals),
        "diagonals_format": "compact",
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "tridiag_op_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorTridiag"] = tf_linalg_linear_operator_tridiag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorTridiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorTridiag'.")

check_valid('tf.linalg.LinearOperatorTridiag', generated_inputs['tf.linalg.LinearOperatorTridiag'], lib="tf", suffix=0)
