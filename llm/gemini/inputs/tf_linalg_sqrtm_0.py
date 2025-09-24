
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_sqrtm_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 matrix
    input_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.float32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex64 matrix
    input_matrix = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], dtype=np.complex64)
    input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.complex64)
    input_dict = {"input": input_tensor, "name": "complex_sqrtm"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger float64 matrix
    input_matrix = np.array([[4.0, 1.0], [6.0, 5.0]], dtype=np.float64)
    input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.float64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex128 matrix
    input_matrix = np.array([[4.0 + 0j, 1.0 + 0j], [6.0 + 0j, 5.0 + 0j]], dtype=np.complex128)
    input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.complex128)
    input_dict = {"input": input_tensor, "name": "complex128_sqrtm"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch of matrices
    input_matrix = np.array([[[1.0, 0.0], [0.0, 1.0]], [[4.0, 0.0], [0.0, 9.0]]], dtype=np.float32)
    input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.float32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor float32
    input_matrix = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]], dtype=np.float32)
    input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.float32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2x2 float32 matrix
    input_matrix = np.array([[2.0, 3.0], [1.0, 4.0]], dtype=np.float32)
    input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.float32)
    input_dict = {"input": input_tensor, "name": "another_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple batches of matrices (float64)
    input_matrix = np.array([[[1.0, 0.0], [0.0, 1.0]], [[4.0, 1.0], [2.0, 5.0]]], dtype=np.float64)
    input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.float64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Batch of complex64 matrices
    input_matrix = np.array([[[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], [[2.0 + 0j, 0.0 + 0j], [0.0 + 0j, 2.0 + 0j]]], dtype=np.complex64)
    input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.complex64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Simple float64
    input_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.float64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.sqrtm"] = tf_linalg_sqrtm_inputs()

for i in range(len(generated_inputs["tf.linalg.sqrtm"])):
    generated_inputs["tf.linalg.sqrtm"][i]["input"] = generated_inputs["tf.linalg.sqrtm"][i]["input"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.sqrtm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.sqrtm'.")

check_valid('tf.linalg.sqrtm', generated_inputs['tf.linalg.sqrtm'], lib="tf", suffix=0)
