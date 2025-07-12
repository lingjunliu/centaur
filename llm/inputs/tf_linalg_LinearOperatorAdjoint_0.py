
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatoradjoint_inputs():
    list_of_inputs = []

    # Input 1
    operator_np = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_1"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operator_np = np.array([[1 + 1j, 2], [3, 4 - 1j]], dtype=np.complex64)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_2"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    operator_np = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "adjoint_op_3"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    operator_np = np.array([[1., 2.], [2., 1.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_4"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    operator_np = np.array([[1., 0.], [0., -1.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_5"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    operator_np = np.array([[5., 0.], [0., 1.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "adjoint_op_6"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    operator_np = np.array([[1j, 0.], [0., 1j]], dtype=np.complex64)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_7"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    op_array = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(op_array)
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_8"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    operator_np = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "adjoint_op_9"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    operator_np = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = False
    name = "adjoint_op_10"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11 - Different matrix size
    operator_np = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = False
    name = "adjoint_op_11"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12 - Rank 3 tensor
    operator_np = np.random.rand(2, 3, 4).astype(np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = False
    name = "adjoint_op_12"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13 - Empty Matrix, changed to a 2d array to prevent Value Error
    operator_np = np.array([[]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = False
    name = "adjoint_op_13"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14 - 1x1 matrix
    operator_np = np.array([[5.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(operator_np)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "adjoint_op_14"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
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
