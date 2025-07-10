
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_inv_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 matrix
    input_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"input": tf.constant(input_matrix).numpy(), "adjoint": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 matrix with adjoint
    input_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"input": tf.constant(input_matrix).numpy(), "adjoint": True, "name": "adjoint_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: half precision matrix
    input_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict = {"input": tf.constant(input_matrix).numpy(), "adjoint": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64 matrix
    input_matrix = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], dtype=np.complex64)
    input_dict = {"input": tf.constant(input_matrix).numpy(), "adjoint": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex128 matrix with adjoint
    input_matrix = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], dtype=np.complex128)
    input_dict = {"input": tf.constant(input_matrix).numpy(), "adjoint": True, "name": "complex_adjoint"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch of matrices (float32)
    input_matrix = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"input": tf.constant(input_matrix).numpy(), "adjoint": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger matrix (float32)
    input_matrix = np.array([[1.0, 2.0, 3.0], [0.0, 5.0, 6.0], [7.0, 0.0, 9.0]], dtype=np.float32)
    input_dict = {"input": tf.constant(input_matrix).numpy(), "adjoint": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Matrix with negative values (float32)
    input_matrix = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {"input": tf.constant(input_matrix).numpy(), "adjoint": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batch of complex matrices (complex64)
    input_matrix = np.array([[[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], [[5.0, 6.0 + 1j], [7.0 - 1j, 8.0]]], dtype=np.complex64)
    input_dict = {"input": tf.constant(input_matrix).numpy(), "adjoint": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensor of matrices (float32)
    input_matrix = np.random.rand(2, 3, 2, 2).astype(np.float32)
    input_dict = {"input": tf.constant(input_matrix).numpy(), "adjoint": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.inv"] = tf_linalg_inv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.inv'.")

check_valid('tf.linalg.inv', generated_inputs['tf.linalg.inv'], lib="tf", suffix=0)
