
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_expm_inputs():
    list_of_inputs = []

    # Input 1: float32, 2x2 matrix
    input_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_tensor = tf.convert_to_tensor(input_matrix)
    input_dict = {"input": input_tensor, "name": "expm_float32_2x2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 3x3 matrix
    input_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    input_tensor = tf.convert_to_tensor(input_matrix)
    input_dict = {"input": input_tensor, "name": "expm_float64_3x3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, 2x2 matrix
    input_matrix = np.array([[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]], dtype=np.complex64)
    input_tensor = tf.convert_to_tensor(input_matrix)
    input_dict = {"input": input_tensor, "name": "expm_complex64_2x2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, 3x3 matrix
    input_matrix = np.array([[1.0j, 0.0, 0.0], [0.0, 1.0j, 0.0], [0.0, 0.0, 1.0j]], dtype=np.complex128)
    input_tensor = tf.convert_to_tensor(input_matrix)
    input_dict = {"input": input_tensor, "name": "expm_complex128_3x3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, batched matrices
    input_matrix = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = tf.convert_to_tensor(input_matrix)
    input_dict = {"input": input_tensor, "name": "expm_float32_batched"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 7: float32, Identity matrix
    input_matrix = np.eye(3, dtype=np.float32)
    input_tensor = tf.convert_to_tensor(input_matrix)
    input_dict = {"input": input_tensor, "name": "expm_float32_identity"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, negative values
    input_matrix = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    input_tensor = tf.convert_to_tensor(input_matrix)
    input_dict = {"input": input_tensor, "name": "expm_float64_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64, zero matrix
    input_matrix = np.array([[0.0j, 0.0j], [0.0j, 0.0j]], dtype=np.complex64)
    input_tensor = tf.convert_to_tensor(input_matrix)
    input_dict = {"input": input_tensor, "name": "expm_complex64_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_linalg_expm_inputs()
for i in range(len(inputs)):
  inputs[i]["input"] = inputs[i]["input"].numpy()
generated_inputs["tf.linalg.expm"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.expm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.expm'.")

check_valid('tf.linalg.expm', generated_inputs['tf.linalg.expm'], lib="tf", suffix=0)
