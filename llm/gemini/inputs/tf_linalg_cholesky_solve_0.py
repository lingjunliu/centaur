
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_cholesky_solve_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    chol = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0]], dtype=np.float32)
    name = "cholesky_solve_1"
    input_dict = {"chol": tf.convert_to_tensor(chol).numpy(), "rhs": tf.convert_to_tensor(rhs).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch case
    chol = np.array([[[2.0, 0.0], [1.0, 3.0]], [[1.0, 0.0], [0.5, 2.0]]], dtype=np.float32)
    rhs = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    name = "cholesky_solve_2"
    input_dict = {"chol": tf.convert_to_tensor(chol).numpy(), "rhs": tf.convert_to_tensor(rhs).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger matrix
    chol = np.array([[4.0, 0.0, 0.0], [2.0, 3.0, 0.0], [1.0, 1.0, 2.0]], dtype=np.float64)
    rhs = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    name = "cholesky_solve_3"
    input_dict = {"chol": tf.convert_to_tensor(chol).numpy(), "rhs": tf.convert_to_tensor(rhs).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple RHS
    chol = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    rhs = np.array([[1.0, 4.0], [2.0, 5.0]], dtype=np.float32)
    name = "cholesky_solve_4"
    input_dict = {"chol": tf.convert_to_tensor(chol).numpy(), "rhs": tf.convert_to_tensor(rhs).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch case with multiple RHS
    chol = np.array([[[2.0, 0.0], [1.0, 3.0]], [[1.0, 0.0], [0.5, 2.0]]], dtype=np.float32)
    rhs = np.array([[[1.0, 4.0], [2.0, 5.0]], [[3.0, 6.0], [4.0, 7.0]]], dtype=np.float32)
    name = "cholesky_solve_5"
    input_dict = {"chol": tf.convert_to_tensor(chol).numpy(), "rhs": tf.convert_to_tensor(rhs).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger batch and RHS
    chol = np.random.rand(5, 3, 3).astype(np.float64)
    for i in range(5):
        chol[i] = np.tril(chol[i])
    rhs = np.random.rand(5, 3, 4).astype(np.float64)
    name = "cholesky_solve_6"
    input_dict = {"chol": tf.convert_to_tensor(chol).numpy(), "rhs": tf.convert_to_tensor(rhs).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: 1D matrix
    chol = np.array([[5.0]], dtype=np.float32)
    rhs = np.array([[2.0]], dtype=np.float32)
    name = "cholesky_solve_7"
    input_dict = {"chol": tf.convert_to_tensor(chol).numpy(), "rhs": tf.convert_to_tensor(rhs).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Larger RHS dim
    chol = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    rhs = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    name = "cholesky_solve_8"
    input_dict = {"chol": tf.convert_to_tensor(chol).numpy(), "rhs": tf.convert_to_tensor(rhs).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another batch case with different dimensions
    chol = np.random.rand(2, 4, 4).astype(np.float32)
    for i in range(2):
        chol[i] = np.tril(chol[i])
    rhs = np.random.rand(2, 4, 2).astype(np.float32)
    name = "cholesky_solve_9"
    input_dict = {"chol": tf.convert_to_tensor(chol).numpy(), "rhs": tf.convert_to_tensor(rhs).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64
    chol = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float64)
    rhs = np.array([[1.0], [2.0]], dtype=np.float64)
    name = "cholesky_solve_10"
    input_dict = {"chol": tf.convert_to_tensor(chol).numpy(), "rhs": tf.convert_to_tensor(rhs).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.cholesky_solve"] = tf_linalg_cholesky_solve_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.cholesky_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.cholesky_solve'.")

check_valid('tf.linalg.cholesky_solve', generated_inputs['tf.linalg.cholesky_solve'], lib="tf", suffix=0)
