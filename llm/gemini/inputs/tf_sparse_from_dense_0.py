
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_from_dense_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with some zeros
    tensor = np.array([0, 1, 0, 2, 0], dtype=np.int32)
    name = "sparse_tensor_1"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with zeros
    tensor = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.int64)
    name = "sparse_tensor_2"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array with zeros
    tensor = np.array([[[0, 1], [2, 0]], [[3, 0], [0, 4]]], dtype=np.float32)
    name = "sparse_tensor_3"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with negative values and zeros
    tensor = np.array([0, -1, 0, 2, -3, 0], dtype=np.int16)
    name = "sparse_tensor_4"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with mixed positive, negative and zero values
    tensor = np.array([[-1, 0, 2], [0, -3, 4]], dtype=np.float64)
    name = "sparse_tensor_5"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with only one non-zero value
    tensor = np.array([0, 0, 0, 5, 0, 0], dtype=np.int8)
    name = "sparse_tensor_6"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty 2D array
    tensor = np.array([[]], dtype=np.int32)
    name = "sparse_tensor_7"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with a different dtype
    tensor = np.array([[0, 1.5, 0], [2.5, 0, 3.5]], dtype=np.float16)
    name = "sparse_tensor_8"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with more complex structure and different dtype
    tensor = np.array([[[0, 1.2], [2.3, 0]], [[3.4, 0], [0, 4.5]]], dtype=np.float64)
    name = "sparse_tensor_9"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array of booleans
    tensor = np.array([False, True, False, True], dtype=np.bool_)
    name = "sparse_tensor_10"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.from_dense"] = tf_sparse_from_dense_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sparse.from_dense' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.from_dense'.")

check_valid('tf.sparse.from_dense', generated_inputs['tf.sparse.from_dense'], lib="tf", suffix=0)
