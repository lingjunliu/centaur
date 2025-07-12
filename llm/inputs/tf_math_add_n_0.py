
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_add_n_inputs():
    list_of_inputs = []

    # Input 1: Basic addition of two matrices
    inputs = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Addition of three matrices with a name
    inputs = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": "sum_of_matrices"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Addition of matrices with negative values
    inputs = [np.array([[-1, 2], [3, -4]]), np.array([[5, -6], [-7, 8]])]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Addition of matrices with zero values
    inputs = [np.array([[0, 2], [3, 0]]), np.array([[5, 0], [0, 8]])]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Addition of 3D tensors
    inputs = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Addition of 1D tensors (vectors)
    inputs = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Addition of a single tensor (should return the tensor itself)
    inputs = [np.array([[1, 2], [3, 4]])]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger matrices
    inputs = [np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), np.array([[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Adding 2 1D tensors with negative values
    inputs = [np.array([-1, 2, -3]), np.array([4, -5, 6])]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": "negative_sum"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Addition of a list of identical tensors
    inputs = [np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]])]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D tensors
    inputs = [np.random.rand(2, 3, 4, 5), np.random.rand(2, 3, 4, 5)]
    input_dict = {"inputs": [tf.convert_to_tensor(x, dtype=tf.float32) for x in inputs], "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: float64
    inputs = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.add_n"] = tf_math_add_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.add_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.add_n'.")

check_valid('tf.math.add_n', generated_inputs['tf.math.add_n'], lib="tf", suffix=0)
