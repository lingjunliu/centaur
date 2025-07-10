
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_matvec_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([5, 6], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "adjoint_a": False, "a_is_sparse": False, "b_is_sparse": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Transpose a
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([5, 6], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": True, "adjoint_a": False, "a_is_sparse": False, "b_is_sparse": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Adjoint a (complex)
    a = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    b = np.array([5+5j, 6+6j], dtype=np.complex64)
    input_dict = {"a": a, "b": b, "transpose_a": False, "adjoint_a": True, "a_is_sparse": False, "b_is_sparse": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Sparse a (float32)
    a = np.array([[0, 2], [3, 0]], dtype=np.float32)
    b = np.array([5, 6], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "adjoint_a": False, "a_is_sparse": True, "b_is_sparse": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrix and vector
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float64)
    b = np.array([13, 14, 15, 16], dtype=np.float64)
    input_dict = {"a": a, "b": b, "transpose_a": False, "adjoint_a": False, "a_is_sparse": False, "b_is_sparse": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values
    a = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    b = np.array([-5, 6], dtype=np.int32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "adjoint_a": False, "a_is_sparse": False, "b_is_sparse": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: 3D tensor a, 2D tensor b
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    b = np.array([[9, 10], [11, 12]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "adjoint_a": False, "a_is_sparse": False, "b_is_sparse": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int32 type
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([5, 6], dtype=np.int32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "adjoint_a": False, "a_is_sparse": False, "b_is_sparse": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex128 type
    a = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    b = np.array([5+5j, 6+6j], dtype=np.complex128)
    input_dict = {"a": a, "b": b, "transpose_a": False, "adjoint_a": False, "a_is_sparse": False, "b_is_sparse": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: a_is_sparse with float32 type
    a = np.array([[0, 2], [3, 0]], dtype=np.float32)
    b = np.array([5, 6], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "adjoint_a": False, "a_is_sparse": True, "b_is_sparse": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.matvec"] = tf_linalg_matvec_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.matvec' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.matvec'.")

check_valid('tf.linalg.matvec', generated_inputs['tf.linalg.matvec'], lib="tf", suffix=0)
