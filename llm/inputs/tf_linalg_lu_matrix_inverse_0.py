
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_lu_matrix_inverse_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([[3., 4], [1, 2]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    validate_args = False
    name = None
    input_dict = {"lower_upper": lu_np, "perm": perm_np, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[7., 8], [3, 4]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    validate_args = True
    name = "inverse"
    input_dict = {"lower_upper": lu_np, "perm": perm_np, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple matrices
    x = np.array([[[3., 4], [1, 2]], [[7., 8], [3, 4]]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    validate_args = False
    name = "another_inverse"
    input_dict = {"lower_upper": lu_np, "perm": perm_np, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4: 3x3 matrix
    x = np.array([[1., 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    validate_args = True
    name = None
    input_dict = {"lower_upper": lu_np, "perm": perm_np, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple 3x3 matrices
    x = np.array([[[1., 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    validate_args = False
    name = "multi_3x3"
    input_dict = {"lower_upper": lu_np, "perm": perm_np, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: different dtype
    x = np.array([[3., 4], [1, 2]], dtype=np.float64)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    validate_args = True
    name = None
    input_dict = {"lower_upper": lu_np, "perm": perm_np, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1x1 matrix
    x = np.array([[5.]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    validate_args = False
    name = "1x1"
    input_dict = {"lower_upper": lu_np, "perm": perm_np, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: multiple 1x1 matrices
    x = np.array([[[5.]], [[2.]]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    validate_args = True
    name = None
    input_dict = {"lower_upper": lu_np, "perm": perm_np, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger Matrix
    x = np.random.rand(5, 5).astype(np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    validate_args = False
    name = "larger_matrix"
    input_dict = {"lower_upper": lu_np, "perm": perm_np, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Removed rectangular matrix due to error.

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.lu_matrix_inverse"] = tf_linalg_lu_matrix_inverse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.lu_matrix_inverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.lu_matrix_inverse'.")

check_valid('tf.linalg.lu_matrix_inverse', generated_inputs['tf.linalg.lu_matrix_inverse'], lib="tf", suffix=0)
