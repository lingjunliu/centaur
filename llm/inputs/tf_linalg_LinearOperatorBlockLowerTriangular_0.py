
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorblocklowertriangular_inputs():
    list_of_inputs = []

    # Input 1
    operator_0 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    operators = [[operator_0]]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "test_op_1"

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
    operator_0 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    operator_1 = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 6.], [7., 8.]], dtype=np.float32))
    operators = [[operator_0]] # modified - only one row
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "test_op_2"

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
    operator_0 = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    operators = [[operator_0]]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "test_op_3"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: batched operators
    matrix_44 = np.random.normal(size=[2, 4, 4]).astype(np.float32)
    operator_44 = tf.linalg.LinearOperatorFullMatrix(matrix_44)
    operators = [[operator_44]]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "test_op_4"
    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    operator_0 = tf.linalg.LinearOperatorFullMatrix(np.array([[-1., 2.], [3., -4.]], dtype=np.float32))
    operators = [[operator_0]]
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "test_op_7"

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
generated_inputs["tf.linalg.LinearOperatorBlockLowerTriangular"] = tf_linalg_linearoperatorblocklowertriangular_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorBlockLowerTriangular' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorBlockLowerTriangular'.")

check_valid('tf.linalg.LinearOperatorBlockLowerTriangular', generated_inputs['tf.linalg.LinearOperatorBlockLowerTriangular'], lib="tf", suffix=0)
