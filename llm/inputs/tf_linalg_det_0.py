
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_det_inputs():
    list_of_inputs = []

    # Input 1: float32, 2x2
    input_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"input": tf.constant(input_matrix), "name": "det_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 3x3
    input_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    input_dict = {"input": tf.constant(input_matrix), "name": "det_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, 2x2
    input_matrix = np.array([[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]], dtype=np.complex64)
    input_dict = {"input": tf.constant(input_matrix), "name": "det_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, 3x3
    input_matrix = np.array([[1.0 + 1j, 2.0 + 2j, 3.0], [4.0, 5.0 + 5j, 6.0], [7.0, 8.0, 9.0 + 9j]], dtype=np.complex128)
    input_dict = {"input": tf.constant(input_matrix), "name": "det_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, batch of 2x2
    input_matrix = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"input": tf.constant(input_matrix), "name": "det_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, batch of 3x3
    input_matrix = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]], dtype=np.float64)
    input_dict = {"input": tf.constant(input_matrix), "name": "det_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, 1x1
    input_matrix = np.array([[5.0]], dtype=np.float32)
    input_dict = {"input": tf.constant(input_matrix), "name": "det_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, batch of 1x1
    input_matrix = np.array([[[5.0]], [[6.0]]], dtype=np.float64)
    input_dict = {"input": tf.constant(input_matrix), "name": "det_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, batch of 2x2
    input_matrix = np.array([[[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]], [[5.0 - 1j, 6.0 - 2j], [7.0 - 3j, 8.0 - 4j]]], dtype=np.complex128)
    input_dict = {"input": tf.constant(input_matrix), "name": "det_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_linalg_det_inputs()
for i in range(len(inputs)):
    inputs[i]['input'] = inputs[i]['input'].numpy()
generated_inputs["tf.linalg.det"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.det' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.det'.")

check_valid('tf.linalg.det', generated_inputs['tf.linalg.det'], lib="tf", suffix=0)
