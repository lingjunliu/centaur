
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_matmul_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[5, 6], [7, 8]], dtype=np.int32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "adjoint_a": False, "adjoint_b": False, "a_is_sparse": False, "b_is_sparse": False, "output_type": None, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": True, "transpose_b": False, "adjoint_a": False, "adjoint_b": False, "a_is_sparse": False, "b_is_sparse": False, "output_type": None, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    b = np.array([[5+5j, 6+6j], [7+7j, 8+8j]], dtype=np.complex64)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": True, "adjoint_a": False, "adjoint_b": False, "a_is_sparse": False, "b_is_sparse": False, "output_type": None, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[5, 6], [7, 8]], dtype=np.int32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "adjoint_a": False, "adjoint_b": True, "a_is_sparse": False, "b_is_sparse": False, "output_type": None, "grad_a": False, "grad_b": False, "name": "matmul_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "adjoint_a": False, "adjoint_b": False, "a_is_sparse": False, "b_is_sparse": False, "output_type": None, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([[1, 0], [0, 4]], dtype=np.float32)
    b = np.array([[5, 0], [0, 8]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "adjoint_a": False, "adjoint_b": False, "a_is_sparse": True, "b_is_sparse": True, "output_type": None, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[1, 2, 3]], dtype=np.int32)
    b = np.array([[4], [5], [6]], dtype=np.int32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "adjoint_a": False, "adjoint_b": False, "a_is_sparse": False, "b_is_sparse": False, "output_type": None, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[[1, 2]], [[3, 4]]], dtype=np.int32)
    b = np.array([[[5], [6]], [[7], [8]]], dtype=np.int32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "adjoint_a": False, "adjoint_b": False, "a_is_sparse": False, "b_is_sparse": False, "output_type": None, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[1, 2], [3, 4]], dtype=np.float16)
    b = np.array([[5, 6], [7, 8]], dtype=np.float16)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "adjoint_a": False, "adjoint_b": False, "a_is_sparse": False, "b_is_sparse": False, "output_type": None, "grad_a": True, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    b = np.array([[5, 6], [7, 8]], dtype=np.float64)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "adjoint_a": False, "adjoint_b": False, "a_is_sparse": False, "b_is_sparse": False, "output_type": None, "grad_a": False, "grad_b": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.matmul"] = tf_linalg_matmul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.matmul'.")

check_valid('tf.linalg.matmul', generated_inputs['tf.linalg.matmul'], lib="tf", suffix=0)
