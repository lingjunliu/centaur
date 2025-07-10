
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_lu_reconstruct_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    input_dict = {
        "lower_upper": lu_np,
        "perm": perm_np,
        "validate_args": False,
        "name": "lu_reconstruct_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    input_dict = {
        "lower_upper": lu_np,
        "perm": perm_np,
        "validate_args": True,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A batch of matrices
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    input_dict = {
        "lower_upper": lu_np,
        "perm": perm_np,
        "validate_args": False,
        "name": "batch_lu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger matrix
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    input_dict = {
        "lower_upper": lu_np,
        "perm": perm_np,
        "validate_args": True,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex numbers
    x = np.array([[1.0+1j, 2.0+2j], [3.0+3j, 4.0+4j]], dtype=np.complex64)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    input_dict = {
        "lower_upper": lu_np,
        "perm": perm_np,
        "validate_args": True,
        "name": "complex_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Double precision
    x = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    input_dict = {
        "lower_upper": lu_np,
        "perm": perm_np,
        "validate_args": False,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shaped batch
    x = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 19.0]]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    input_dict = {
        "lower_upper": lu_np,
        "perm": perm_np,
        "validate_args": True,
        "name": "diff_batch"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Identity matrix
    x = np.eye(3, dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    input_dict = {
        "lower_upper": lu_np,
        "perm": perm_np,
        "validate_args": False,
        "name": "identity_matrix"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Matrix with zeros but still invertible
    x = np.array([[1e-5, 1.0], [1.0, 0.0]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    input_dict = {
        "lower_upper": lu_np,
        "perm": perm_np,
        "validate_args": True,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Input 10: Small values
    x = np.array([[1e-4, 1e-3], [1e-2, 1e-1]], dtype=np.float32)
    lu = tf.linalg.lu(x)
    lu_np = lu.lu.numpy()
    perm_np = lu.p.numpy()
    input_dict = {
        "lower_upper": lu_np,
        "perm": perm_np,
        "validate_args": False,
        "name": "small_values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.lu_reconstruct"] = tf_linalg_lu_reconstruct_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.lu_reconstruct' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.lu_reconstruct'.")

check_valid('tf.linalg.lu_reconstruct', generated_inputs['tf.linalg.lu_reconstruct'], lib="tf", suffix=0)
