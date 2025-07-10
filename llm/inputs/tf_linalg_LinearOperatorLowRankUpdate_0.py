
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorlowrankupdate_inputs():
    list_of_inputs = []

    def create_linear_operator_diag(diag, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True):
        return tf.linalg.LinearOperatorDiag(diag=diag, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=True)

    def create_linear_operator_full_matrix(matrix, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True):
        return tf.linalg.LinearOperatorFullMatrix(matrix=matrix, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=True)

    # Input 1
    diag = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    base_operator = create_linear_operator_diag(diag)
    u = np.array([[1.0, 2.0], [-1.0, 3.0], [0.0, 0.0]], dtype=np.float32)
    diag_update = np.array([11.0, 12.0], dtype=np.float32)
    v = np.array([[1.0, 2.0], [-1.0, 3.0], [10.0, 10.0]], dtype=np.float32)
    is_diag_update_positive = True
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "low_rank_update_1"

    input_dict = {
        "base_operator": base_operator,
        "u": u,
        "diag_update": diag_update,
        "v": v,
        "is_diag_update_positive": is_diag_update_positive,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    base_operator = create_linear_operator_full_matrix(matrix)
    u = np.array([[1.0], [1.0]], dtype=np.float32)
    diag_update = np.array([2.0], dtype=np.float32)
    v = np.array([[1.0], [1.0]], dtype=np.float32)
    is_diag_update_positive = False
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "low_rank_update_2"

    input_dict = {
        "base_operator": base_operator,
        "u": u,
        "diag_update": diag_update,
        "v": v,
        "is_diag_update_positive": is_diag_update_positive,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    base_operator = tf.linalg.LinearOperatorScaledIdentity(num_rows=2, multiplier=1.0)
    u = np.array([[1.0], [0.0]], dtype=np.float32)
    diag_update = np.array([-1.0], dtype=np.float32)
    v = np.array([[0.0], [1.0]], dtype=np.float32)
    is_diag_update_positive = None
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "low_rank_update_3"

    input_dict = {
        "base_operator": base_operator,
        "u": u,
        "diag_update": diag_update,
        "v": v,
        "is_diag_update_positive": is_diag_update_positive,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorLowRankUpdate"] = tf_linalg_linearoperatorlowrankupdate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorLowRankUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorLowRankUpdate'.")

check_valid('tf.linalg.LinearOperatorLowRankUpdate', generated_inputs['tf.linalg.LinearOperatorLowRankUpdate'], lib="tf", suffix=0)
