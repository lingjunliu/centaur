
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorcomposition_inputs():
    list_of_inputs = []

    # Input 1
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]]))
    operators = [op1, op2]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "composition_1"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., 1.]]))
    operators = [op1]
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "identity"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    op1_matrix = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    op2_matrix = np.array([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]])

    op1 = tf.linalg.LinearOperatorFullMatrix(op1_matrix)
    op2 = tf.linalg.LinearOperatorFullMatrix(op2_matrix)

    operators = [op1, op2]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = None

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorComposition"] = tf_linalg_linearoperatorcomposition_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorComposition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorComposition'.")

check_valid('tf.linalg.LinearOperatorComposition', generated_inputs['tf.linalg.LinearOperatorComposition'], lib="tf", suffix=0)
