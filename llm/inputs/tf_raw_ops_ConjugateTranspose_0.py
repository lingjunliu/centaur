
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_conjugate_transpose_inputs():
    list_of_inputs = []

    # Input 1, valid
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    perm = np.array([1, 0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": "transpose_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    perm = np.array([0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": "transpose_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    perm = np.array([0, 1, 2], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": "transpose_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    perm = np.array([2, 0, 1], dtype=np.int64)
    input_dict = {"x": x, "perm": perm, "name": "transpose_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    perm = np.array([1, 0], dtype=np.int64)
    input_dict = {"x": x, "perm": perm, "name": "transpose_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    perm = np.array([0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": "transpose_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.complex64)
    perm = np.array([1, 2, 0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": "transpose_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    perm = np.array([1, 0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": "transpose_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    perm = np.array([0, 2, 1], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": "transpose_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    perm = np.array([0], dtype=np.int64)
    input_dict = {"x": x, "perm": perm, "name": "transpose_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ConjugateTranspose"] = tf_raw_ops_conjugate_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ConjugateTranspose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ConjugateTranspose'.")

check_valid('tf.raw_ops.ConjugateTranspose', generated_inputs['tf.raw_ops.ConjugateTranspose'], lib="tf", suffix=0)
