
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Diag_inputs():
    list_of_inputs = []

    # Input 1: Basic integer diagonal
    diagonal = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float diagonal
    diagonal = np.array([1.0, 2.5, 3.7, 4.2], dtype=np.float32)
    input_dict = {"diagonal": diagonal, "name": "diag2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex diagonal
    diagonal = np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex64)
    input_dict = {"diagonal": diagonal, "name": "diag3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int64 diagonal
    diagonal = np.array([1, 2, 3, 4], dtype=np.int64)
    input_dict = {"diagonal": diagonal, "name": "diag4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty diagonal
    diagonal = np.array([], dtype=np.float32)
    input_dict = {"diagonal": diagonal, "name": "diag5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element diagonal
    diagonal = np.array([5], dtype=np.int32)
    input_dict = {"diagonal": diagonal, "name": "diag6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Half type diagonal
    diagonal = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"diagonal": diagonal, "name": "diag7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128 diagonal
    diagonal = np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex128)
    input_dict = {"diagonal": diagonal, "name": "diag10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Diag"] = tf_raw_ops_Diag_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Diag'.")

check_valid('tf.raw_ops.Diag', generated_inputs['tf.raw_ops.Diag'], lib="tf", suffix=0)
