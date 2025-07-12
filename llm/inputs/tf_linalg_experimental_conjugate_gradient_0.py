
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_experimental_conjugate_gradient_inputs():
    list_of_inputs = []

    # Input 1
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]]))
    rhs = tf.constant(np.array([1., 2.]), dtype=tf.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=tf.float32)
    x = tf.constant(np.array([0., 0.]), dtype=tf.float32)
    tol = 1e-05
    max_iter = 20
    name = "cg1"

    input_dict = {
        "operator": operator.to_dense().numpy(),
        "rhs": rhs.numpy(),
        "preconditioner": preconditioner.to_dense().numpy(),
        "x": x.numpy(),
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[4., 1.], [1., 3.]]))
    rhs = tf.constant(np.array([1., 2.]), dtype=tf.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=tf.float32)
    x = tf.constant(np.array([0., 0.]), dtype=tf.float32)
    tol = 1e-03
    max_iter = 50
    name = "cg2"

    input_dict = {
        "operator": operator.to_dense().numpy(),
        "rhs": rhs.numpy(),
        "preconditioner": preconditioner.to_dense().numpy(),
        "x": x.numpy(),
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]]))
    rhs = tf.constant(np.array([1., 2.]), dtype=tf.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=tf.float32)
    x = tf.constant(np.array([0.5, 0.5]), dtype=tf.float32)
    tol = 1e-07
    max_iter = 10
    name = "cg3"

    input_dict = {
        "operator": operator.to_dense().numpy(),
        "rhs": rhs.numpy(),
        "preconditioner": preconditioner.to_dense().numpy(),
        "x": x.numpy(),
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch rhs
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]]))
    rhs = tf.constant(np.array([[1., 2.], [3., 4.]]), dtype=tf.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=tf.float32)
    x = tf.constant(np.array([[0., 0.], [0., 0.]]), dtype=tf.float32)
    tol = 1e-05
    max_iter = 20
    name = "cg4"

    input_dict = {
        "operator": operator.to_dense().numpy(),
        "rhs": rhs.numpy(),
        "preconditioner": preconditioner.to_dense().numpy(),
        "x": x.numpy(),
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 2.], [2., 3.]]))
    rhs = tf.constant(np.array([3., 5.]), dtype=tf.float32)
    preconditioner = tf.linalg.LinearOperatorFullMatrix(np.array([[0.2, 0.], [0., 0.3]]))
    x = tf.constant(np.array([0., 0.]), dtype=tf.float32)
    tol = 1e-06
    max_iter = 30
    name = "cg5"

    input_dict = {
        "operator": operator.to_dense().numpy(),
        "rhs": rhs.numpy(),
        "preconditioner": preconditioner.to_dense().numpy(),
        "x": x.numpy(),
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: different initial guess
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]]))
    rhs = tf.constant(np.array([1., 2.]), dtype=tf.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=tf.float32)
    x = tf.constant(np.array([1., 1.]), dtype=tf.float32)
    tol = 1e-05
    max_iter = 20
    name = "cg6"

    input_dict = {
        "operator": operator.to_dense().numpy(),
        "rhs": rhs.numpy(),
        "preconditioner": preconditioner.to_dense().numpy(),
        "x": x.numpy(),
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[3., 1.], [1., 4.]]))
    rhs = tf.constant(np.array([2., 3.]), dtype=tf.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=tf.float32)
    x = tf.constant(np.array([0., 0.]), dtype=tf.float32)
    tol = 1e-04
    max_iter = 40
    name = "cg7"

    input_dict = {
        "operator": operator.to_dense().numpy(),
        "rhs": rhs.numpy(),
        "preconditioner": preconditioner.to_dense().numpy(),
        "x": x.numpy(),
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[10., 1.], [1., 5.]]))
    rhs = tf.constant(np.array([5., 3.]), dtype=tf.float32)
    preconditioner = tf.linalg.LinearOperatorFullMatrix(np.array([[0.1, 0.], [0., 0.2]]))
    x = tf.constant(np.array([0., 0.]), dtype=tf.float32)
    tol = 1e-08
    max_iter = 60
    name = "cg8"

    input_dict = {
        "operator": operator.to_dense().numpy(),
        "rhs": rhs.numpy(),
        "preconditioner": preconditioner.to_dense().numpy(),
        "x": x.numpy(),
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9 - different dimensions
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]]))
    rhs = tf.constant(np.array([1., 2., 3.]), dtype=tf.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=3, dtype=tf.float32)
    x = tf.constant(np.array([0., 0., 0.]), dtype=tf.float32)
    tol = 1e-05
    max_iter = 20
    name = "cg9"

    input_dict = {
        "operator": operator.to_dense().numpy(),
        "rhs": rhs.numpy(),
        "preconditioner": preconditioner.to_dense().numpy(),
        "x": x.numpy(),
        "tol": tol,
        "max_iter": max_iter,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - higher tolerance and less iterations
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]]))
    rhs = tf.constant(np.array([1., 2.]), dtype=tf.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=tf.float32)
    x = tf.constant(np.array([0., 0.]), dtype=tf.float32)
    tol = 1e-01
    max_iter = 5
    name = "cg10"

    input_dict = {
        "operator": operator.to_dense().numpy(),
        "rhs": rhs.numpy(),
        "preconditioner": preconditioner.to_dense().numpy(),
        "x": x.numpy(),
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
