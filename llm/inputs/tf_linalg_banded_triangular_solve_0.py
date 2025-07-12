
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_banded_triangular_solve_inputs():
    list_of_inputs = []

    # Input 1: Basic lower triangular solve
    bands = np.array([[2., 0., 0.], [1., 2., 0.]]).astype(np.float32)
    rhs = np.array([[1.], [1.], [1.]]).astype(np.float32)
    lower = True
    adjoint = False
    name = "basic_lower"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic upper triangular solve
    bands = np.array([[2., 3., 4.], [0., 2., 3.]]).astype(np.float32)
    rhs = np.array([[1.], [1.], [1.]]).astype(np.float32)
    lower = False
    adjoint = False
    name = "basic_upper"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple RHS vectors
    bands = np.array([[2., 0., 0.], [1., 2., 0.]]).astype(np.float32)
    rhs = np.array([[1., 2.], [1., 2.], [1., 2.]]).astype(np.float32)
    lower = True
    adjoint = False
    name = "multiple_rhs"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of matrices
    bands = np.array([[[2., 0., 0.], [1., 2., 0.]], [[3., 0., 0.], [2., 3., 0.]]]).astype(np.float32)
    rhs = np.array([[1., 1., 1.], [2., 2., 2.]]).astype(np.float32)
    lower = True
    adjoint = False
    name = "batch"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shaped RHS
    bands = np.array([[2., 0.], [1., 2.]]).astype(np.float32)
    rhs = np.array([[1., 2.], [1., 2.]]).astype(np.float32)
    lower = True
    adjoint = False
    name = "different_rhs"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: adjoint=True
    bands = np.array([[2., 0., 0.], [1., 2., 0.]]).astype(np.float32)
    rhs = np.array([[1.], [1.], [1.]]).astype(np.float32)
    lower = True
    adjoint = True
    name = "adjoint_true"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64
    bands = np.array([[2., 0., 0.], [1., 2., 0.]]).astype(np.float64)
    rhs = np.array([[1.], [1.], [1.]]).astype(np.float64)
    lower = True
    adjoint = False
    name = "float64"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Simple upper triangular solve - Modified to ensure invertibility
    bands = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).astype(np.float32)
    rhs = np.array([[1.0], [1.0], [1.0]]).astype(np.float32)
    lower = False
    adjoint = False
    name = "simple_upper"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: Non square rhs
    bands = np.array([[2., 0.], [1., 2.]]).astype(np.float32)
    rhs = np.array([[1., 2.], [1., 3.]]).astype(np.float32)
    lower = True
    adjoint = False
    name = "non_square_rhs"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different bands
    bands = np.array([[3., 0., 0.], [1., 3., 0.]]).astype(np.float32)
    rhs = np.array([[1.], [1.], [1.]]).astype(np.float32)
    lower = True
    adjoint = False
    name = "different_bands"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.banded_triangular_solve"] = tf_linalg_banded_triangular_solve_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.banded_triangular_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.banded_triangular_solve'.")

check_valid('tf.linalg.banded_triangular_solve', generated_inputs['tf.linalg.banded_triangular_solve'], lib="tf", suffix=0)
