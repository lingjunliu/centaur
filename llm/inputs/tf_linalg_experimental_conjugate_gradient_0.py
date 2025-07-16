
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_experimental_conjugate_gradient_inputs():
    list_of_inputs = []

    # Input 1
    matrix1 = np.array([[2., 1.], [1., 2.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix1).to_dense()
    rhs = np.array([1., 2.], dtype=np.float32)
    preconditioner_matrix = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    preconditioner = tf.linalg.LinearOperatorFullMatrix(preconditioner_matrix).to_dense()
    x = np.array([0., 0.], dtype=np.float32)
    tol = 1e-05
    max_iter = 20
    name = "cg1"

    input_dict = {
        "operator": operator,
        "rhs": rhs,
        "preconditioner": preconditioner,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    matrix2 = np.array([[4., 1.], [1., 3.]], dtype=np.float64)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix2).to_dense()
    rhs = np.array([1., 0.], dtype=np.float64)
    preconditioner_matrix = np.array([[1., 0.], [0., 1.]], dtype=np.float64)
    preconditioner = tf.linalg.LinearOperatorFullMatrix(preconditioner_matrix).to_dense()
    x = np.array([0., 0.], dtype=np.float64)
    tol = 1e-08
    max_iter = 50
    name = "cg2"

    input_dict = {
        "operator": operator,
        "rhs": rhs,
        "preconditioner": preconditioner,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    matrix3 = np.array([[2., 0.], [0., 2.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix3).to_dense()
    rhs = np.array([3., 4.], dtype=np.float32)
    preconditioner = None
    x = np.array([1., 1.], dtype=np.float32)
    tol = 1e-04
    max_iter = 10
    name = "cg3"

    input_dict = {
        "operator": operator,
        "rhs": rhs,
        "preconditioner": preconditioner,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    matrix4 = np.array([[5., 2.], [2., 3.]], dtype=np.float64)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix4).to_dense()
    rhs = np.array([5., 7.], dtype=np.float64)
    preconditioner = None
    x = np.array([0., 0.], dtype=np.float64)
    tol = 1e-06
    max_iter = 30
    name = "cg4"

    input_dict = {
        "operator": operator,
        "rhs": rhs,
        "preconditioner": preconditioner,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    matrix5 = np.array([[2., 1.], [1., 2.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix5).to_dense()
    rhs = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    preconditioner_matrix = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    preconditioner = tf.linalg.LinearOperatorFullMatrix(preconditioner_matrix).to_dense()
    x = np.array([[0., 0.], [0., 0.]], dtype=np.float32)
    tol = 1e-05
    max_iter = 20
    name = "cg5"

    input_dict = {
        "operator": operator,
        "rhs": rhs,
        "preconditioner": preconditioner,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    matrix6 = np.array([[4., 1.], [1., 3.]], dtype=np.float64)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix6).to_dense()
    rhs = np.array([[1., 0.], [2., 1.]], dtype=np.float64)
    preconditioner_matrix = np.array([[1., 0.], [0., 1.]], dtype=np.float64)
    preconditioner = tf.linalg.LinearOperatorFullMatrix(preconditioner_matrix).to_dense()
    x = np.array([[0., 0.], [0., 0.]], dtype=np.float64)
    tol = 1e-08
    max_iter = 50
    name = "cg6"

    input_dict = {
        "operator": operator,
        "rhs": rhs,
        "preconditioner": preconditioner,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    matrix7 = np.array([[2., 0.], [0., 2.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix7).to_dense()
    rhs = np.array([[3., 4.], [5., 6.]], dtype=np.float32)
    preconditioner = None
    x = np.array([[1., 1.], [1., 1.]], dtype=np.float32)
    tol = 1e-04
    max_iter = 10
    name = "cg7"

    input_dict = {
        "operator": operator,
        "rhs": rhs,
        "preconditioner": preconditioner,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    matrix8 = np.array([[5., 2.], [2., 3.]], dtype=np.float64)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix8).to_dense()
    rhs = np.array([[5., 7.], [9., 11.]], dtype=np.float64)
    preconditioner = None
    x = np.array([[0., 0.], [0., 0.]], dtype=np.float64)
    tol = 1e-06
    max_iter = 30
    name = "cg8"

    input_dict = {
        "operator": operator,
        "rhs": rhs,
        "preconditioner": preconditioner,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    matrix9 = np.array([[[2., 1.], [1., 2.]]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix9).to_dense()
    rhs = np.array([[[1., 2.]]], dtype=np.float32)
    preconditioner_matrix = np.array([[[1., 0.], [0., 1.]]], dtype=np.float32)
    preconditioner = tf.linalg.LinearOperatorFullMatrix(preconditioner_matrix).to_dense()
    x = np.array([[[0., 0.]]], dtype=np.float32)
    tol = 1e-05
    max_iter = 20
    name = "cg9"

    input_dict = {
        "operator": operator,
        "rhs": rhs,
        "preconditioner": preconditioner,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    matrix10 = np.array([[[4., 1.], [1., 3.]]], dtype=np.float64)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix10).to_dense()
    rhs = np.array([[[1., 0.]]], dtype=np.float64)
    preconditioner_matrix = np.array([[[1., 0.], [0., 1.]]], dtype=np.float64)
    preconditioner = tf.linalg.LinearOperatorFullMatrix(preconditioner_matrix).to_dense()
    x = np.array([[[0., 0.]]], dtype=np.float64)
    tol = 1e-08
    max_iter = 50
    name = "cg10"

    input_dict = {
        "operator": operator,
        "rhs": rhs,
        "preconditioner": preconditioner,
        "x": x,
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.experimental.conjugate_gradient"] = tf_linalg_experimental_conjugate_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.experimental.conjugate_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.experimental.conjugate_gradient'.")

check_valid('tf.linalg.experimental.conjugate_gradient', generated_inputs['tf.linalg.experimental.conjugate_gradient'], lib="tf", suffix=0)
