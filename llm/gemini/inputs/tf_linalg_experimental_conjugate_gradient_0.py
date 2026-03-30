
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_experimental_conjugate_gradient_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]], dtype=np.float32))
    rhs = np.array([1., 2.], dtype=np.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=np.float32)
    x = np.array([0., 0.], dtype=np.float32)
    tol = 1e-05
    max_iter = 20
    name = "conjugate_gradient_1"
    input_dict = {"operator": operator, "rhs": rhs, "preconditioner": preconditioner, "x": x, "tol": tol, "max_iter": max_iter, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No preconditioner
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[4., 1.], [1., 3.]], dtype=np.float32))
    rhs = np.array([2., 3.], dtype=np.float32)
    preconditioner = None
    x = np.array([0., 0.], dtype=np.float32)
    tol = 1e-05
    max_iter = 20
    name = "conjugate_gradient_2"

    input_dict = {"operator": operator, "rhs": rhs, "preconditioner": preconditioner, "x": x, "tol": tol, "max_iter": max_iter, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different initial guess
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]], dtype=np.float32))
    rhs = np.array([1., 2.], dtype=np.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=np.float32)
    x = np.array([0.5, 0.5], dtype=np.float32)
    tol = 1e-05
    max_iter = 20
    name = "conjugate_gradient_3"
    input_dict = {"operator": operator, "rhs": rhs, "preconditioner": preconditioner, "x": x, "tol": tol, "max_iter": max_iter, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tighter tolerance
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]], dtype=np.float32))
    rhs = np.array([1., 2.], dtype=np.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=np.float32)
    x = np.array([0., 0.], dtype=np.float32)
    tol = 1e-08
    max_iter = 20
    name = "conjugate_gradient_4"
    input_dict = {"operator": operator, "rhs": rhs, "preconditioner": preconditioner, "x": x, "tol": tol, "max_iter": max_iter, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Fewer iterations
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[2., 1.], [1., 2.]], dtype=np.float32))
    rhs = np.array([1., 2.], dtype=np.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=np.float32)
    x = np.array([0., 0.], dtype=np.float32)
    tol = 1e-05
    max_iter = 5
    name = "conjugate_gradient_5"
    input_dict = {"operator": operator, "rhs": rhs, "preconditioner": preconditioner, "x": x, "tol": tol, "max_iter": max_iter, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3x3 matrix
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[4., 1., 1.], [1., 3., 1.], [1., 1., 2.]], dtype=np.float32))
    rhs = np.array([1., 2., 3.], dtype=np.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=3, dtype=np.float32)
    x = np.array([0., 0., 0.], dtype=np.float32)
    tol = 1e-05
    max_iter = 20
    name = "conjugate_gradient_6"
    input_dict = {"operator": operator, "rhs": rhs, "preconditioner": preconditioner, "x": x, "tol": tol, "max_iter": max_iter, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Batch of matrices
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[[2., 1.], [1., 2.]], [[3., 1.], [1., 3.]]], dtype=np.float32))
    rhs = np.array([[1., 2.], [2., 3.]], dtype=np.float32)
    preconditioner = tf.linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[2], dtype=np.float32)
    x = np.array([[0., 0.], [0., 0.]], dtype=np.float32)
    tol = 1e-05
    max_iter = 20
    name = "conjugate_gradient_7"
    input_dict = {"operator": operator, "rhs": rhs, "preconditioner": preconditioner, "x": x, "tol": tol, "max_iter": max_iter, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.experimental.conjugate_gradient"] = tf_linalg_experimental_conjugate_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.experimental.conjugate_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.experimental.conjugate_gradient'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.experimental.conjugate_gradient', generated_inputs['tf.linalg.experimental.conjugate_gradient'], lib="tf", suffix=0)
