
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_lu_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 matrix
    input_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"input": tf.constant(input_matrix), "output_idx_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 matrix with negative values
    input_matrix = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    input_dict = {"input": tf.constant(input_matrix), "output_idx_type": tf.int64, "name": "lu_decomposition"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half (float16) matrix
    input_matrix = np.array([[5.0, 2.0], [1.0, 9.0]], dtype=np.float16)
    input_dict = {"input": tf.constant(input_matrix), "output_idx_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64 matrix
    input_matrix = np.array([[1 + 1j, 2 - 1j], [3 + 0j, 4 - 2j]], dtype=np.complex64)
    input_dict = {"input": tf.constant(input_matrix), "output_idx_type": tf.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex128 matrix
    input_matrix = np.array([[1 + 1j, 2 - 1j], [3 + 0j, 4 - 2j]], dtype=np.complex128)
    input_dict = {"input": tf.constant(input_matrix), "output_idx_type": tf.int64, "name": "complex_lu"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.lu"] = tf_linalg_lu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.lu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.lu'.")

check_valid('tf.linalg.lu', generated_inputs['tf.linalg.lu'], lib="tf", suffix=0)
