
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_norm_inputs():
    list_of_inputs = []

    # Input 1: Vector, default ord, no axis
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"tensor": tensor, "ord": 'euclidean', "axis": None, "keepdims": False, "name": "norm_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix, Frobenius norm, no axis
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "ord": 'fro', "axis": None, "keepdims": False, "name": "norm_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of vectors, axis=1
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "ord": 'euclidean', "axis": [1], "keepdims": False, "name": "norm_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of matrices, axis=(0, 1), keepdims=True
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"tensor": tensor, "ord": 'fro', "axis": [0, 1], "keepdims": True, "name": "norm_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Vector, 1-norm, no axis
    tensor = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    input_dict = {"tensor": tensor, "ord": '1', "axis": None, "keepdims": False, "name": "norm_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Matrix, inf-norm, axis=(0,1)
    tensor = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "ord": 'np.inf', "axis": (0, 1), "keepdims": False, "name": "norm_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor, axis=0
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"tensor": tensor, "ord": 'euclidean', "axis": [0], "keepdims": False, "name": "norm_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Vector with complex numbers
    tensor = np.array([1.0 + 1j, 2.0 - 2j, 3.0 + 0j], dtype=np.complex64)
    input_dict = {"tensor": tensor, "ord": 'euclidean', "axis": None, "keepdims": False, "name": "norm_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: axis=-1
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "ord": 'euclidean', "axis": [-1], "keepdims": False, "name": "norm_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batch of matrices, 1-norm, axis=(0, 1), keepdims=True
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"tensor": tensor, "ord": '1', "axis": [0, 1], "keepdims": True, "name": "norm_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.norm_2"] = tf_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.norm_2'.")

check_valid('tf.norm', generated_inputs['tf.norm_2'], lib="tf", suffix=2)
