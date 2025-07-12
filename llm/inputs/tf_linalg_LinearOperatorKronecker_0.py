
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_LinearOperatorKronecker_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 matrices
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]]))
    operators = [op1, op2]
    input_dict = {"operators": operators, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": None, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of matrices with flags set
    matrix1 = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    matrix2 = np.array([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]])
    op1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    op2 = tf.linalg.LinearOperatorFullMatrix(matrix2)

    operators = [op1, op2]
    input_dict = {"operators": operators, "is_non_singular": True, "is_self_adjoint": False, "is_positive_definite": True, "is_square": True, "name": "kronecker_product_batch"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single matrix, treated as a list of one
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    operators = [op1]
    input_dict = {"operators": operators, "is_non_singular": False, "is_self_adjoint": True, "is_positive_definite": False, "is_square": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different sized matrices (but square)
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6., 7.], [7., 8., 9.], [10., 11., 12.]]))
    operators = [op1, op2]
    input_dict = {"operators": operators, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": None, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Matrices with negative values
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[-1., 2.], [3., -4.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., -6.], [-7., 8.]]))
    operators = [op1, op2]
    input_dict = {"operators": operators, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": None, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Identity matrices
    op1 = tf.linalg.LinearOperatorIdentity(num_rows=2)
    op2 = tf.linalg.LinearOperatorIdentity(num_rows=3)
    operators = [op1, op2]
    input_dict = {"operators": operators, "is_non_singular": True, "is_self_adjoint": True, "is_positive_definite": True, "is_square": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7:  A mix of LinearOperatorIdentity and LinearOperatorFullMatrix
    op1 = tf.linalg.LinearOperatorIdentity(num_rows=2)
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., -6.], [-7., 8.]]))
    operators = [op1, op2]
    input_dict = {"operators": operators, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": None, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger matrices
    op1 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(5, 5))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.random.rand(4, 4))
    operators = [op1, op2]
    input_dict = {"operators": operators, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": None, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batched identity with explicit flag
    op1 = tf.linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[2])
    op2 = tf.linalg.LinearOperatorIdentity(num_rows=3, batch_shape=[2])
    operators = [op1, op2]
    input_dict = {"operators": operators, "is_non_singular": True, "is_self_adjoint": True, "is_positive_definite": True, "is_square": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  List of three matrices
    op1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]]))
    op2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]]))
    op3 = tf.linalg.LinearOperatorFullMatrix(np.array([[9., 10.], [11., 12.]]))
    operators = [op1, op2, op3]
    input_dict = {"operators": operators, "is_non_singular": None, "is_self_adjoint": None, "is_positive_definite": None, "is_square": None, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorKronecker"] = tf_linalg_LinearOperatorKronecker_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorKronecker' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorKronecker'.")

check_valid('tf.linalg.LinearOperatorKronecker', generated_inputs['tf.linalg.LinearOperatorKronecker'], lib="tf", suffix=0)
