
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_experimental_conjugate_gradient_inputs():
    list_of_inputs = []

    # Input 1
    operator = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    rhs = np.array([1.0, 1.0], dtype=np.float32)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'cg1'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': None, 'x': None, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operator = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    rhs = np.array([1.0, 2.0], dtype=np.float32)
    if np.linalg.det(operator) != 0:
        preconditioner = np.linalg.inv(operator)
        preconditioner = tf.linalg.LinearOperatorFullMatrix(tf.constant(preconditioner))
    else:
        preconditioner = None

    x = np.array([0.0, 0.0], dtype=np.float32)
    tol = 1e-06
    max_iter = 50
    name = 'cg2'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': preconditioner, 'x': tf.constant(x), 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 - Batched rhs
    operator = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    rhs = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'cg3'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - Initial guess
    operator = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    rhs = np.array([1.0, 1.0], dtype=np.float32)
    preconditioner = None
    x = np.array([0.5, 0.5], dtype=np.float32)
    tol = 1e-05
    max_iter = 20
    name = 'cg4'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': preconditioner, 'x': tf.constant(x), 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - Higher tolerance
    operator = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    rhs = np.array([1.0, 1.0], dtype=np.float32)
    preconditioner = None
    x = None
    tol = 1e-03
    max_iter = 20
    name = 'cg5'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - More iterations
    operator = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    rhs = np.array([1.0, 1.0], dtype=np.float32)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 100
    name = 'cg6'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Different matrix
    operator = np.array([[5.0, -2.0], [-2.0, 1.0]], dtype=np.float32)
    rhs = np.array([1.0, 1.0], dtype=np.float32)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'cg7'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8 - Complex numbers
    operator = np.array([[2.0+1j, 1.0], [1.0, 2.0-1j]], dtype=np.complex64)
    rhs = np.array([1.0+1j, 1.0-1j], dtype=np.complex64)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'cg8'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Identity preconditioner (should be same as None)
    operator = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    rhs = np.array([1.0, 1.0], dtype=np.float32)
    preconditioner = np.eye(2, dtype=np.float32)
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'cg9'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': None, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Float64
    operator = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    rhs = np.array([1.0, 1.0], dtype=np.float64)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = 20
    name = 'cg10'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': max_iter, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - max_iter as numpy integer
    operator = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    rhs = np.array([1.0, 1.0], dtype=np.float32)
    preconditioner = None
    x = None
    tol = 1e-05
    max_iter = np.int32(20)
    name = 'cg11'
    input_dict = {'operator': tf.linalg.LinearOperatorFullMatrix(tf.constant(operator)), 'rhs': tf.constant(rhs), 'preconditioner': preconditioner, 'x': x, 'tol': tol, 'max_iter': np.int32(max_iter), 'name': name}
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
