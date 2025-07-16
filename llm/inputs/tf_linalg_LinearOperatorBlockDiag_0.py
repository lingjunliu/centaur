
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorblockdiag_inputs():
    list_of_inputs = []

    # Input 1
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]]))
    operators = [op1.to_dense().numpy(), op2.to_dense().numpy()]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "block_diag_1"

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
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 0.], [0., 2.]]))
    operators = [op1.to_dense().numpy(), op2.to_dense().numpy()]
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "block_diag_2"

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
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1.], [2.], [3.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[4., 5.]]))
    operators = [op1.to_dense().numpy(), op2.to_dense().numpy()]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = False
    name = "block_diag_3"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    operators = [op1.to_dense().numpy()]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "block_diag_4"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]))
    operators = [op1.to_dense().numpy(), op2.to_dense().numpy()]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "block_diag_5"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]]))
    operators = [op1.to_dense().numpy(), op2.to_dense().numpy()]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
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

    # Input 7
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]]))
    op3 = tf.linalg.LinearOperatorFullMatrix(np.array([[9., 10.], [11., 12.]]))
    operators = [op1.to_dense().numpy(), op2.to_dense().numpy(), op3.to_dense().numpy()]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "block_diag_7"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple batches, square = False
    op1 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 3, 2, 1))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(2, 3, 1, 2))
    operators = [op1.to_dense().numpy(), op2.to_dense().numpy()]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    name = "block_diag_8"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty name
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]]))
    operators = [op1.to_dense().numpy(), op2.to_dense().numpy()]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = ""

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All Nones
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]]))
    operators = [op1.to_dense().numpy(), op2.to_dense().numpy()]
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
    
    generated_inputs["tf.linalg.LinearOperatorBlockDiag"] = list_of_inputs
    return list_of_inputs

generated_inputs = {}
tf_linalg_linearoperatorblockdiag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorBlockDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorBlockDiag'.")

check_valid('tf.linalg.LinearOperatorBlockDiag', generated_inputs['tf.linalg.LinearOperatorBlockDiag'], lib="tf", suffix=0)
