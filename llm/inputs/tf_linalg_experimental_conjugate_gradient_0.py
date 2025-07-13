
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_conjugate_gradient_inputs():
    list_of_inputs = []

    # Helper function to create a LinearOperator
    def make_linear_operator(matrix):
        return tf.linalg.LinearOperatorFullMatrix(matrix)

    # Input 1
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    rhs = np.array([1., 2.], dtype=np.float32)
    M = np.array([[0.3, 0], [0, 0.3]], dtype=np.float32)
    x = np.array([0., 0.], dtype=np.float32)
    tol = 1e-5
    max_iter = 20
    name = "cg_1"
    input_dict = {
        "operator": make_linear_operator(A).to_dense(),
        "rhs": rhs,
        "preconditioner": make_linear_operator(M).to_dense(),
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    A = np.array([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]], dtype=np.float32)
    rhs = np.array([1., 0., 1.], dtype=np.float32)
    M = np.diag([0.5, 0.5, 0.5]).astype(np.float32)
    x = np.array([0., 0., 0.], dtype=np.float32)
    tol = 1e-6
    max_iter = 30
    name = "cg_2"
    input_dict = {
        "operator": make_linear_operator(A).to_dense(),
        "rhs": rhs,
        "preconditioner": make_linear_operator(M).to_dense(),
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 (no preconditioner)
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float64)
    rhs = np.array([1., 2.], dtype=np.float64)
    x = np.array([0., 0.], dtype=np.float64)
    tol = 1e-7
    max_iter = 40
    name = "cg_3"
    input_dict = {
        "operator": make_linear_operator(A).to_dense(),
        "rhs": rhs,
        "preconditioner": None,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (different initial guess)
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    rhs = np.array([1., 2.], dtype=np.float32)
    M = np.array([[0.3, 0], [0, 0.3]], dtype=np.float32)
    x = np.array([1., 1.], dtype=np.float32)
    tol = 1e-5
    max_iter = 20
    name = "cg_4"
    input_dict = {
        "operator": make_linear_operator(A).to_dense(),
        "rhs": rhs,
        "preconditioner": make_linear_operator(M).to_dense(),
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (higher tolerance)
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    rhs = np.array([1., 2.], dtype=np.float32)
    M = np.array([[0.3, 0], [0, 0.3]], dtype=np.float32)
    x = np.array([0., 0.], dtype=np.float32)
    tol = 1e-2
    max_iter = 20
    name = "cg_5"
    input_dict = {
        "operator": make_linear_operator(A).to_dense(),
        "rhs": rhs,
        "preconditioner": make_linear_operator(M).to_dense(),
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.experimental.conjugate_gradient"] = tf_linalg_conjugate_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.experimental.conjugate_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.experimental.conjugate_gradient'.")

check_valid('tf.linalg.experimental.conjugate_gradient', generated_inputs['tf.linalg.experimental.conjugate_gradient'], lib="tf", suffix=0)
