
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_matrix_transpose_inputs():
    list_of_inputs = []

    # Input 1
    a = tf.constant([[1, 2, 3], [4, 5, 6]]).numpy()
    name = "transpose_1"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = tf.constant([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]).numpy()
    name = "transpose_2"
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    name = "transpose_3"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = tf.constant([[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], [[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]]]).numpy()
    name = "transpose_4"
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]).numpy()
    name = "transpose_5"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = tf.constant([[[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], [[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]]], [[[9 + 9j, 10 + 10j], [11 + 11j, 12 + 12j]], [[13 + 13j, 14 + 14j], [15 + 15j, 16 + 16j]]]]).numpy()
    name = "transpose_6"
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = tf.constant([[-1, -2, -3], [-4, -5, -6]]).numpy()
    name = "transpose_7"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = tf.constant([[-1 - 1j, -2 - 2j], [-3 - 3j, -4 - 4j]]).numpy()
    name = "transpose_8"
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.random.rand(2, 5, 7)
    name = "transpose_9"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = (np.random.rand(3, 4) + 1j * np.random.rand(3, 4))
    name = "transpose_10"
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.matrix_transpose"] = tf_linalg_matrix_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.matrix_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.matrix_transpose'.")

check_valid('tf.linalg.matrix_transpose', generated_inputs['tf.linalg.matrix_transpose'], lib="tf", suffix=0)
