
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D array, default permutation
    a = np.array([[1, 2, 3], [4, 5, 6]])
    perm = [1, 0]
    conjugate = False
    name = "transpose_1"
    input_dict = {"a": a, "perm": perm, "conjugate": conjugate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, custom permutation
    a = np.array([[1, 2, 3], [4, 5, 6]])
    perm = [1, 0]
    conjugate = False
    name = "transpose_2"
    input_dict = {"a": a, "perm": perm, "conjugate": conjugate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, custom permutation
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    perm = [0, 2, 1]
    conjugate = False
    name = "transpose_3"
    input_dict = {"a": a, "perm": perm, "conjugate": conjugate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, different permutation
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    perm = [2, 0, 1]
    conjugate = False
    name = "transpose_4"
    input_dict = {"a": a, "perm": perm, "conjugate": conjugate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array
    a = np.array([1, 2, 3, 4, 5])
    perm = [0]
    conjugate = False
    name = "transpose_5"
    input_dict = {"a": a, "perm": perm, "conjugate": conjugate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D array
    a = np.random.rand(2, 3, 4, 5)
    perm = [3, 1, 0, 2]
    conjugate = False
    name = "transpose_6"
    input_dict = {"a": a, "perm": perm, "conjugate": conjugate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex 2D array, conjugate=True
    a = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    perm = [1, 0]
    conjugate = True
    name = "transpose_7"
    input_dict = {"a": a, "perm": perm, "conjugate": conjugate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Complex 3D array, conjugate=True, custom perm
    a = np.array([[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], [[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]]])
    perm = [0, 2, 1]
    conjugate = True
    name = "transpose_8"
    input_dict = {"a": a, "perm": perm, "conjugate": conjugate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher dimension array with int64 dtype
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    perm = [2, 0, 1]
    conjugate = False
    name = "transpose_9"
    input_dict = {"a": a, "perm": perm, "conjugate": conjugate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Default transpose with int64
    a = np.arange(6, dtype=np.int64).reshape((2, 3))
    perm = [1, 0]
    conjugate = False
    name = "transpose_10"
    input_dict = {"a": a, "perm": perm, "conjugate": conjugate, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.transpose"] = tf_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.transpose'.")

check_valid('tf.transpose', generated_inputs['tf.transpose'], lib="tf", suffix=0)
