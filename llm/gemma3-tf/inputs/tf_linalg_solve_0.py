
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_solve_inputs():
    list_of_inputs = []

    matrix1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs1 = np.array([[5.0], [11.0]], dtype=np.float32)
    adjoint1 = False
    name1 = "solve_example_1"

    input_dict1 = {
        "matrix": matrix1,
        "rhs": rhs1,
        "adjoint": adjoint1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    matrix2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    rhs2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    adjoint2 = False
    name2 = "solve_example_2"

    input_dict2 = {
        "matrix": matrix2,
        "rhs": rhs2,
        "adjoint": adjoint2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    matrix3 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    rhs3 = np.array([[[9.0], [10.0]], [[11.0], [12.0]]], dtype=np.float32)
    adjoint3 = False
    name3 = "solve_example_3"

    input_dict3 = {
        "matrix": matrix3,
        "rhs": rhs3,
        "adjoint": adjoint3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    matrix4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rhs4 = np.array([[1.0], [2.0]], dtype=np.float64)
    adjoint4 = True
    name4 = "solve_example_4"

    input_dict4 = {
        "matrix": matrix4,
        "rhs": rhs4,
        "adjoint": adjoint4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    matrix5 = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    rhs5 = np.array([[5.0], [8.0]], dtype=np.float32)
    adjoint5 = False
    name5 = "solve_example_5"

    input_dict5 = {
        "matrix": matrix5,
        "rhs": rhs5,
        "adjoint": adjoint5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["tf.linalg.solve"] = tf_linalg_solve_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.solve'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.solve', generated_inputs['tf.linalg.solve'], lib="tf", suffix=0)
