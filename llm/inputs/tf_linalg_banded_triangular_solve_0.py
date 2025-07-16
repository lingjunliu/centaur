
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_banded_triangular_solve_inputs():
    list_of_inputs = []

    # Input 1
    bands = np.array([[2., 0., 0.], [1., 2., 0.]], dtype=np.float32)
    rhs = np.array([[1.], [1.], [1.]], dtype=np.float32)
    lower = True
    adjoint = False
    name = "test1"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    bands = np.array([[2., 3., 0., 0.], [0., 2., 3., 0.]], dtype=np.float32)
    rhs = np.array([[1.], [1.], [1.], [1.]], dtype=np.float32)
    lower = False
    adjoint = False
    name = "test2"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    bands = np.array([[-1., 0., 0.], [0., -2., 0.]], dtype=np.float32)
    rhs = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    lower = False
    adjoint = True
    name = "test3"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    bands = np.array([[1., 0.], [3., 4.]], dtype=np.float32)
    rhs = np.array([[5.], [6.]], dtype=np.float32)
    lower = True
    adjoint = True
    name = "test4"
    input_dict = {"bands": bands, "rhs": rhs, "lower": lower, "adjoint": adjoint, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    bands = np.array([[2., 3., 4.], [0.1, 2., 3.]], dtype=np.float32)
    rhs = np.array([7., 8., 9.], dtype=np.float32)
    lower = True
    adjoint = False
    name = "test5"
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
