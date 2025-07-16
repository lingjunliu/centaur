
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperator_kronecker_inputs():
    list_of_inputs = []

    def create_linear_operator(matrix):
      return tf.linalg.LinearOperatorFullMatrix(np.array(matrix).astype(np.float32))

    # Input 1
    op1 = create_linear_operator([[1.0, 2.0], [3.0, 4.0]])
    op2 = create_linear_operator([[5.0, 6.0], [7.0, 8.0]])
    operators = [op1, op2]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "kronecker_1"

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
    op1 = create_linear_operator([[1.0, 0.0], [0.0, 1.0]])
    op2 = create_linear_operator([[0.0, 1.0], [1.0, 0.0]])
    operators = [op1, op2]
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "kronecker_2"

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
    op1 = create_linear_operator([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    op2 = create_linear_operator([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    operators = [op1, op2]
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "kronecker_3"

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
    op1 = create_linear_operator([[1.0]])
    op2 = create_linear_operator([[2.0]])
    op3 = create_linear_operator([[3.0]])
    operators = [op1, op2, op3]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "kronecker_4"

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
    op1 = create_linear_operator([[1.0, 0.0], [0.0, 1.0]])
    op2 = create_linear_operator([[1.0, 2.0], [3.0, 4.0]])
    operators = [op1, op2]
    is_non_singular = True
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

    # Input 6
    op1 = create_linear_operator([[1.0, 1.0], [1.0, 1.0]])
    op2 = create_linear_operator([[2.0, 2.0], [2.0, 2.0]])
    operators = [op1, op2]
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "kronecker_6"

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
    op1 = create_linear_operator([[1.0]])
    operators = [op1]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "kronecker_7"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More diverse values
    op1 = create_linear_operator([[1.5, -2.2], [3.7, 4.1]])
    op2 = create_linear_operator([[-5.3, 6.8], [7.9, -8.4]])
    operators = [op1, op2]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "kronecker_8"
    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 9: Identity matrices
    op1 = tf.linalg.LinearOperatorIdentity(num_rows=2)
    op2 = tf.linalg.LinearOperatorIdentity(num_rows=3)
    operators = [op1, op2]
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "kronecker_9"

    input_dict = {
        "operators": operators,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 10 : Complex number
    op1 = create_linear_operator([[1.0 + 1j, 2.0], [3.0, 4.0]])
    op2 = create_linear_operator([[5.0, 6.0], [7.0, 8.0+2j]])
    operators = [op1, op2]
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "kronecker_10"

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
generated_inputs["tf.linalg.LinearOperatorKronecker"] = tf_linalg_linearoperator_kronecker_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorKronecker' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorKronecker'.")

check_valid('tf.linalg.LinearOperatorKronecker', generated_inputs['tf.linalg.LinearOperatorKronecker'], lib="tf", suffix=0)
