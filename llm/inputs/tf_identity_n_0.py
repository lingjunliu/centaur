
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_identity_n_inputs():
    list_of_inputs = []

    # Input 1: List of single tensor
    tensor1 = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    input_dict = {"input": [tensor1], "name": "single_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of multiple tensors
    tensor2 = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.float32))
    tensor3 = tf.constant(np.array([5, 6, 7, 8], dtype=np.int64))
    input_dict = {"input": [tensor2, tensor3], "name": "multiple_tensors"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of tensors with different shapes
    tensor4 = tf.constant(np.array(10, dtype=np.int32))
    tensor5 = tf.constant(np.array([1, 2], dtype=np.float64))
    input_dict = {"input": [tensor4, tensor5], "name": "different_shapes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of tensors with different data types
    tensor6 = tf.constant(np.array([True, False], dtype=np.bool_))
    tensor7 = tf.constant(np.array([b"hello", b"world"]))
    input_dict = {"input": [tensor6, tensor7], "name": "different_types"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of 3D tensors
    tensor8 = tf.constant(np.random.rand(2, 3, 4), dtype=np.float32)
    tensor9 = tf.constant(np.random.randint(0, 10, size=(3, 2, 5)), dtype=np.int32)
    input_dict = {"input": [tensor8, tensor9], "name": "3d_tensors"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List of tensors with large values
    tensor10 = tf.constant(np.array([1e9, 2e9], dtype=np.float32))
    input_dict = {"input": [tensor10], "name": "large_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List of tensors with negative values
    tensor11 = tf.constant(np.array([-1, -2, -3], dtype=np.int32))
    input_dict = {"input": [tensor11], "name": "negative_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of tensors with a specific name
    tensor12 = tf.constant(np.array([1, 2, 3]), name="my_tensor")
    input_dict = {"input": [tensor12], "name": "specific_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Combination of different types, shapes, and names
    tensor13 = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.float32), name="tensor_a")
    tensor14 = tf.constant(np.array([5, 6, 7]), dtype=np.int64, name="tensor_b")
    tensor15 = tf.constant(True, dtype=np.bool_, name="tensor_c")

    input_dict = {"input": [tensor13, tensor14, tensor15], "name": "complex_example"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.identity_n"] = tf_identity_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.identity_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.identity_n'.")

check_valid('tf.identity_n', generated_inputs['tf.identity_n'], lib="tf", suffix=0)
