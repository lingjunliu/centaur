
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linear_operator_composition_inputs():
    list_of_inputs = []

    # Input 1
    operator1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    operator2 = tf.linalg.LinearOperatorFullMatrix(np.array([[5.0, 6.0], [7.0, 8.0]]))
    operators = [operator1, operator2]
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
    operator1 = tf.linalg.LinearOperatorScaledIdentity(3, 2.0)
    operator2 = tf.linalg.LinearOperatorFullMatrix(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    operators = [operator1, operator2]
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "composition_2"

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
    operator1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    operators = [operator1]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "composition_3"

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
    matrix1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    matrix2 = np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]])
    operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
    operators = [operator1, operator2]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "composition_4"

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
    operator1 = tf.linalg.LinearOperatorScaledIdentity(2, 2.0)
    operator2 = tf.linalg.LinearOperatorFullMatrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    operators = [operator1, operator2]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "composition_5"

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
    operator1 = tf.linalg.LinearOperatorFullMatrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    operator2 = tf.linalg.LinearOperatorFullMatrix(np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]))
    operators = [operator1, operator2]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    name = "composition_6"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch operators
    batch_matrix1 = np.random.rand(2, 3, 4, 5)
    batch_matrix2 = np.random.rand(2, 3, 5, 6)
    operator1 = tf.linalg.LinearOperatorFullMatrix(batch_matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(batch_matrix2)
    operators = [operator1, operator2]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    name = "batch_composition"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    identity_op = tf.linalg.LinearOperatorIdentity(num_rows=5)
    operators = [identity_op]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "identity_composition"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex batch operators
    batch_matrix1 = np.random.rand(2, 1, 4, 5)
    batch_matrix2 = np.random.rand(1, 3, 5, 6)
    operator1 = tf.linalg.LinearOperatorFullMatrix(batch_matrix1)
    operator2 = tf.linalg.LinearOperatorFullMatrix(batch_matrix2)
    operators = [operator1, operator2]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    name = "complex_batch_composition"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    scalar_identity = tf.linalg.LinearOperatorScaledIdentity(4, 1.0)
    matrix_op = tf.linalg.LinearOperatorFullMatrix(np.eye(4))
    operators = [scalar_identity, matrix_op]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "scalar_matrix_composition"

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
generated_inputs["tf.linalg.LinearOperatorComposition"] = tf_linalg_linear_operator_composition_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorComposition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorComposition'.")

check_valid('tf.linalg.LinearOperatorComposition', generated_inputs['tf.linalg.LinearOperatorComposition'], lib="tf", suffix=0)
