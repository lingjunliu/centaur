
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_LinearOperatorLowRankUpdate_inputs():
    list_of_inputs = []

    # Input 1
    diag = np.array([1., 2., 3.], dtype=np.float32)
    base_operator = tf.linalg.LinearOperatorDiag(diag=diag, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True)
    u = np.array([[1., 2.], [-1., 3.], [0., 0.]], dtype=np.float32)
    diag_update = np.array([11., 12.], dtype=np.float32)
    v = np.array([[1., 2.], [-1., 3.], [10., 10.]], dtype=np.float32)
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
    base_operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32), is_non_singular=True, is_self_adjoint=False, is_positive_definite=False)
    u = np.array([[1.], [1.]], dtype=np.float32)
    diag_update = np.array([2.], dtype=np.float32)
    v = np.array([[1.], [1.]], dtype=np.float32)
    is_diag_update_positive = False
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
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
    base_operator = tf.linalg.LinearOperatorIdentity(num_rows=5, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True)
    u = np.array([[1., 0.], [0., 1.], [1., 1.], [0., 0.], [1., 0.]], dtype=np.float32)
    diag_update = np.array([1., 1.], dtype=np.float32)
    v = np.array([[0., 1.], [1., 0.], [0., 0.], [1., 1.], [0., 1.]], dtype=np.float32)
    is_diag_update_positive = None
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
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

    # Input 4
    base_operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 0.], [0., 2.]], dtype=np.float32), is_non_singular=True, is_self_adjoint=True, is_positive_definite=True)
    u = np.array([[1.], [2.]], dtype=np.float32)
    diag_update = np.array([1.], dtype=np.float32)
    v = np.array([[4.], [3.]], dtype=np.float32)
    is_diag_update_positive = True
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "low_rank_update_4"

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

    # Input 5
    diag = np.array([1., 1., 1., 1.], dtype=np.float32)
    base_operator = tf.linalg.LinearOperatorDiag(diag=diag, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True)
    u = np.array([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=np.float32)
    diag_update = np.array([-1., 1.], dtype=np.float32)
    v = np.array([[0., 1.], [1., 0.], [0., 0.], [1., 1.]], dtype=np.float32)
    is_diag_update_positive = False
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = False
    is_square = True
    name = "low_rank_update_5"

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

     # Input 6: Rank-1 update with a FullMatrix base operator
    base_operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32), is_non_singular=True, is_self_adjoint=False, is_positive_definite=False)
    u = np.array([[1.], [2.]], dtype=np.float32)
    diag_update = np.array([3.], dtype=np.float32)
    v = np.array([[4.], [5.]], dtype=np.float32)
    is_diag_update_positive = True
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "low_rank_update_6"
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

    # Input 7:  is_square = False
    base_operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32), is_non_singular=True, is_self_adjoint=False, is_positive_definite=False, is_square=False)
    u = np.array([[1.], [2.]], dtype=np.float32)
    diag_update = np.array([3.], dtype=np.float32)
    v = np.array([[4.], [5.], [6.]], dtype=np.float32)
    is_diag_update_positive = True
    is_non_singular = None
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    name = "low_rank_update_7"
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

    # Input 8:  Different dtype
    diag = np.array([1., 2., 3.], dtype=np.float64)
    base_operator = tf.linalg.LinearOperatorDiag(diag=diag, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True)
    u = np.array([[1., 2.], [-1., 3.], [0., 0.]], dtype=np.float64)
    diag_update = np.array([11., 12.], dtype=np.float64)
    v = np.array([[1., 2.], [-1., 3.], [10., 10.]], dtype=np.float64)
    is_diag_update_positive = True
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "low_rank_update_8"
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

    # Input 9: batched base_operator
    diag = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    base_operator = tf.linalg.LinearOperatorDiag(diag=diag, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True)
    u = np.array([[[1., 2.], [-1., 3.], [0., 0.]], [[1., 2.], [-1., 3.], [0., 0.]]], dtype=np.float32)
    diag_update = np.array([[11., 12.], [13., 14.]], dtype=np.float32)
    v = np.array([[[1., 2.], [-1., 3.], [10., 10.]], [[1., 2.], [-1., 3.], [10., 10.]]], dtype=np.float32)
    is_diag_update_positive = True
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "low_rank_update_9"

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

    # Input 10: u and v different shapes, but compatible with base_operator shape
    base_operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32), is_non_singular=True, is_self_adjoint=False, is_positive_definite=False, is_square=False)
    u = np.array([[1.], [2.]], dtype=np.float32)
    diag_update = np.array([3.], dtype=np.float32)
    v = np.array([[4.], [5.], [6.]], dtype=np.float32)
    is_diag_update_positive = True
    is_non_singular = None
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    name = "low_rank_update_10"
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
generated_inputs["tf.linalg.LinearOperatorLowRankUpdate"] = tf_linalg_LinearOperatorLowRankUpdate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorLowRankUpdate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorLowRankUpdate'.")

check_valid('tf.linalg.LinearOperatorLowRankUpdate', generated_inputs['tf.linalg.LinearOperatorLowRankUpdate'], lib="tf", suffix=0)
