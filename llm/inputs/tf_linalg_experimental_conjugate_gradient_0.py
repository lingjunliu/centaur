
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_conjugate_gradient_inputs():
    list_of_inputs = []

    def create_linear_operator(matrix):
        return tf.linalg.LinearOperatorFullMatrix(matrix)

    # Input 1
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    rhs = np.array([1., 2.], dtype=np.float32)
    operator = create_linear_operator(A)
    rhs = tf.constant(rhs)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'cg1'
    input_dict = {'operator': operator, 'rhs': rhs, 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    A = np.array([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]], dtype=np.float32)
    rhs = np.array([1., 0., 1.], dtype=np.float32)
    operator = create_linear_operator(A)
    rhs = tf.constant(rhs)
    M = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
    preconditioner = create_linear_operator(M)
    x = np.array([0., 0., 0.], dtype=np.float32)
    x = tf.constant(x)
    tol = 1e-06
    max_iter = 50
    name = 'cg2'
    input_dict = {'operator': operator, 'rhs': rhs, 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batched rhs
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    rhs = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    operator = create_linear_operator(A)
    rhs = tf.constant(rhs)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'cg3'
    input_dict = {'operator': operator, 'rhs': rhs, 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Initial guess
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    rhs = np.array([1., 2.], dtype=np.float32)
    operator = create_linear_operator(A)
    rhs = tf.constant(rhs)
    preconditioner = None
    x = np.array([0.1, 0.2], dtype=np.float32)
    x = tf.constant(x)
    tol = 1e-05
    max_iter = 20
    name = 'cg4'
    input_dict = {'operator': operator, 'rhs': rhs, 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different tolerance
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    rhs = np.array([1., 2.], dtype=np.float32)
    operator = create_linear_operator(A)
    rhs = tf.constant(rhs)
    preconditioner = None
    x = None
    tol = 1e-03
    max_iter = 20
    name = 'cg5'
    input_dict = {'operator': operator, 'rhs': rhs, 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: Different max_iter
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    rhs = np.array([1., 2.], dtype=np.float32)
    operator = create_linear_operator(A)
    rhs = tf.constant(rhs)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 5
    name = 'cg6'
    input_dict = {'operator': operator, 'rhs': rhs, 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D matrix, 2D rhs
    A = np.array([[[4., 1.], [1., 3.]], [[2, 1], [1, 5]]], dtype=np.float32)
    rhs = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(A)
    rhs = tf.constant(rhs)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'cg7'
    input_dict = {'operator': operator, 'rhs': rhs, 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Identity Preconditioner
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    rhs = np.array([1., 2.], dtype=np.float32)
    operator = create_linear_operator(A)
    rhs = tf.constant(rhs)
    M = np.eye(2, dtype=np.float32)
    preconditioner = create_linear_operator(M)
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'cg8'
    input_dict = {'operator': operator, 'rhs': rhs, 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger system
    A = np.random.rand(10, 10).astype(np.float32)
    A = A @ A.T  # Make it positive definite
    rhs = np.random.rand(10).astype(np.float32)
    operator = create_linear_operator(A)
    rhs = tf.constant(rhs)
    preconditioner = None
    x = None
    tol = 1e-06
    max_iter = 100
    name = 'cg9'
    input_dict = {'operator': operator, 'rhs': rhs, 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different name
    A = np.array([[4., 1.], [1., 3.]], dtype=np.float32)
    rhs = np.array([1., 2.], dtype=np.float32)
    operator = create_linear_operator(A)
    rhs = tf.constant(rhs)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'another_cg'
    input_dict = {'operator': operator, 'rhs': rhs, 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
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
