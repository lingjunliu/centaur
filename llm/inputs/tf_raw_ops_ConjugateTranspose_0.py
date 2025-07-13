
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_conjugate_transpose_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D tensor
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    perm = np.array([1, 0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D tensor with different permutation
    x = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    perm = np.array([2, 0, 1], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor (effectively no transpose)
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    perm = np.array([0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor with int64 permutation
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    perm = np.array([1, 0], dtype=np.int64)
    input_dict = {"x": x, "perm": perm, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger 2D tensor
    x = np.random.rand(5, 5).astype(np.complex64)
    perm = np.array([1, 0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D tensor
    x = np.random.rand(2, 3, 4, 5).astype(np.complex64)
    perm = np.array([3, 1, 0, 2], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Simple 2D tensor, different name
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    perm = np.array([1, 0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": "my_transpose"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128 type
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    perm = np.array([1, 0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D complex128 with int64 perm
    x = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex128)
    perm = np.array([2, 0, 1], dtype=np.int64)
    input_dict = {"x": x, "perm": perm, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Larger tensor
    x = np.random.rand(2, 5, 3).astype(np.complex64)
    perm = np.array([1, 2, 0], dtype=np.int32)
    input_dict = {"x": x, "perm": perm, "name": None}
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
