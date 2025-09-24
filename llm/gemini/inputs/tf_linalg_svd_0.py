
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_svd_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 matrix
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    full_matrices = False
    compute_uv = True
    name = "svd_1"
    input_dict = {"tensor": tensor, "full_matrices": full_matrices, "compute_uv": compute_uv, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No UV computation
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    full_matrices = False
    compute_uv = False
    name = "svd_2"
    input_dict = {"tensor": tensor, "full_matrices": full_matrices, "compute_uv": compute_uv, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Full matrices
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    full_matrices = True
    compute_uv = True
    name = "svd_3"
    input_dict = {"tensor": tensor, "full_matrices": full_matrices, "compute_uv": compute_uv, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rectangular matrix (M > N)
    tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    full_matrices = False
    compute_uv = True
    name = "svd_4"
    input_dict = {"tensor": tensor, "full_matrices": full_matrices, "compute_uv": compute_uv, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrix (M < N)
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    full_matrices = False
    compute_uv = True
    name = "svd_5"
    input_dict = {"tensor": tensor, "full_matrices": full_matrices, "compute_uv": compute_uv, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    full_matrices = False
    compute_uv = True
    name = "svd_6"
    input_dict = {"tensor": tensor, "full_matrices": full_matrices, "compute_uv": compute_uv, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Matrix with negative values
    tensor = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    full_matrices = False
    compute_uv = True
    name = "svd_7"
    input_dict = {"tensor": tensor, "full_matrices": full_matrices, "compute_uv": compute_uv, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different dtype (float64)
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    full_matrices = False
    compute_uv = True
    name = "svd_8"
    input_dict = {"tensor": tensor, "full_matrices": full_matrices, "compute_uv": compute_uv, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Identity matrix
    tensor = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    full_matrices = False
    compute_uv = True
    name = "svd_9"
    input_dict = {"tensor": tensor, "full_matrices": full_matrices, "compute_uv": compute_uv, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero matrix
    tensor = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    full_matrices = False
    compute_uv = True
    name = "svd_10"
    input_dict = {"tensor": tensor, "full_matrices": full_matrices, "compute_uv": compute_uv, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.svd"] = tf_linalg_svd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.svd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.svd'.")

check_valid('tf.linalg.svd', generated_inputs['tf.linalg.svd'], lib="tf", suffix=0)
