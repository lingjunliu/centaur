
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatoradjoint_inputs():
    list_of_inputs = []

    # Input 1
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    input_dict = {"operator": operator, "is_non_singular": True, "is_self_adjoint": False, "is_positive_definite": False, "is_square": True, "name": "op1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1. + 1j, 2.], [3., 4. - 1j]], dtype=np.complex64))
    input_dict = {"operator": operator, "is_non_singular": False, "is_self_adjoint": False, "is_positive_definite": None, "is_square": True, "name": "op2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 0.], [0., 5.]], dtype=np.float32))
    input_dict = {"operator": operator, "is_non_singular": True, "is_self_adjoint": True, "is_positive_definite": True, "is_square": True, "name": "op3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    input_dict = {"operator": operator, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": None, "name": "op4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32))
    input_dict = {"operator": operator, "is_non_singular": False, "is_self_adjoint": False, "is_positive_definite": False, "is_square": True, "name": "op5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., -1.]], dtype=np.float32))
    input_dict = {"operator": operator, "is_non_singular": True, "is_self_adjoint": True, "is_positive_definite": False, "is_square": True, "name": "op6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]], dtype=np.float32))
    input_dict = {"operator": operator, "is_non_singular": True, "is_self_adjoint": True, "is_positive_definite": True, "is_square": True, "name": "op7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1. + 0j, 0.], [0., 1. + 0j]], dtype=np.complex64))
    input_dict = {"operator": operator, "is_non_singular": True, "is_self_adjoint": True, "is_positive_definite": True, "is_square": True, "name": "op8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[0., 1.], [1., 0.]], dtype=np.float32))
    input_dict = {"operator": operator, "is_non_singular": True, "is_self_adjoint": True, "is_positive_definite": False, "is_square": True, "name": "op9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., 0.]], dtype=np.float32))
    input_dict = {"operator": operator, "is_non_singular": False, "is_self_adjoint": True, "is_positive_definite": False, "is_square": True, "name": "op10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorAdjoint"] = tf_linalg_linearoperatoradjoint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorAdjoint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorAdjoint'.")

check_valid('tf.linalg.LinearOperatorAdjoint', generated_inputs['tf.linalg.LinearOperatorAdjoint'], lib="tf", suffix=0)
