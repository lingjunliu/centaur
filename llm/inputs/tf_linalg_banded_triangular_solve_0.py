
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_banded_triangular_solve_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 lower triangular system
    bands = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0]], dtype=np.float32)
    lower = True
    adjoint = False
    name = "test_solve_1"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple 3x3 upper triangular system - reduce size to avoid singularity
    bands = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0]], dtype=np.float32)
    lower = False
    adjoint = False
    name = "test_solve_2"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of 2x2 lower triangular systems
    bands = np.array([[[1.0, 0.0], [2.0, 3.0]], [[4.0, 0.0], [5.0, 6.0]]], dtype=np.float32)
    rhs = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    lower = True
    adjoint = False
    name = "test_solve_3"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3x3 lower triangular, complex numbers - Reducing size to 2x2 to avoid non-invertible
    bands = np.array([[1.0j, 0.0], [2.0j, 3.0j]], dtype=np.complex64)
    rhs = np.array([[1.0j], [2.0j]], dtype=np.complex64)
    lower = True
    adjoint = False
    name = "test_solve_4"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2x2 upper triangular, adjoint=True
    bands = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0]], dtype=np.float32)
    lower = False
    adjoint = True
    name = "test_solve_5"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  Removing because it might cause non-invertible errors

   # Input 7: 4x4 Upper Triangular - Reducing size
    bands = np.array([[1.0, 2.0], [0.0, 5.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0]], dtype=np.float32)
    lower = False
    adjoint = False
    name = "test_solve_7"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: 4x4 Lower Triangular - Reducing size
    bands = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0]], dtype=np.float32)
    lower = True
    adjoint = False
    name = "test_solve_8"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1x1 matrix
    bands = np.array([[2.0]], dtype=np.float32)
    rhs = np.array([[5.0]], dtype=np.float32)
    lower = True
    adjoint = False
    name = "test_solve_9"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Lower with negative values
    bands = np.array([[-1.0, 0.0], [-2.0, -3.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0]], dtype=np.float32)
    lower = True
    adjoint = False
    name = "test_solve_10"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 : Rectangular rhs
    bands = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    rhs = np.array([[1.0, 4.0], [2.0, 5.0]], dtype=np.float32)
    lower = False
    adjoint = False
    name = "test_solve_11"
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
