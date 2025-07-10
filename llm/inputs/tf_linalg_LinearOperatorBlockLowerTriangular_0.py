
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_LinearOperatorBlockLowerTriangular_inputs():
    list_of_inputs = []

    # Input 1
    op_00 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    operators = [[op_00]]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "op1"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    op_00 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    op_10 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 0.], [0., 5.]], dtype=np.float32))
    op_11 = tf.linalg.LinearOperatorFullMatrix(np.array([[0., 2.], [2., 0.]], dtype=np.float32))
    operators = [[op_00], [op_10, op_11]]
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "op2"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    op_00 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    operators = [[op_00]]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "op3"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4: Batch operators
    matrix_44 = np.random.normal(size=[2, 3, 4, 4]).astype(np.float32)
    operator_44 = tf.linalg.LinearOperatorFullMatrix(matrix_44)
    operators = [[operator_44]]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "op4"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1x1 block
    op_00 = tf.linalg.LinearOperatorFullMatrix(np.array([[5.]], dtype=np.float32))
    operators = [[op_00]]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "op6"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: With LinearOperatorLowerTriangular
    op_00 = tf.linalg.LinearOperatorLowerTriangular(np.array([[1., 0.], [3., 4.]], dtype=np.float32))
    operators = [[op_00]]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "op7"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: With LinearOperatorDiag
    op_00 = tf.linalg.LinearOperatorDiag(np.array([1., 4.], dtype=np.float32))
    operators = [[op_00]]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "op8"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    op_00 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    op_10 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    op_11 = tf.linalg.LinearOperatorLowerTriangular(np.array([[5., 0.], [7., 8.]], dtype=np.float32))
    operators = [[op_00], [op_10, op_11]]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "op9"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: all False flags
    op_00 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    operators = [[op_00]]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    name = "op10"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    matrix_44 = np.random.normal(size=[4, 4]).astype(np.float32)
    operator_44 = tf.linalg.LinearOperatorFullMatrix(matrix_44)
    matrix_54 = np.random.normal(size=[5, 4]).astype(np.float32)
    operator_54 = tf.linalg.LinearOperatorFullMatrix(matrix_54)
    matrix_55 = np.random.normal(size=[5, 5]).astype(np.float32)
    operator_55 = tf.linalg.LinearOperatorFullMatrix(matrix_55)
    operators = [[operator_44], [operator_54, operator_55]]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "op11"
    input_dict = {"operators": operators, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorBlockLowerTriangular"] = tf_linalg_LinearOperatorBlockLowerTriangular_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorBlockLowerTriangular' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorBlockLowerTriangular'.")

check_valid('tf.linalg.LinearOperatorBlockLowerTriangular', generated_inputs['tf.linalg.LinearOperatorBlockLowerTriangular'], lib="tf", suffix=0)
