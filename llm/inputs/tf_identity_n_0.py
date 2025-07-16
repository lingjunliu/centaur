
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_identity_n_inputs():
    list_of_inputs = []

    # Input 1: List of a single scalar tensor
    tensor1 = tf.constant(5.0)
    input_dict = {"input": [tensor1], "name": "single_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List with a single 1D tensor
    tensor2 = tf.constant([1, 2, 3], dtype=tf.int32)
    input_dict = {"input": [tensor2], "name": "single_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List with a single 2D tensor
    tensor3 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    input_dict = {"input": [tensor3], "name": "single_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List with two tensors of different shapes
    tensor4 = tf.constant([1, 2, 3])
    tensor5 = tf.constant([[4, 5], [6, 7]])
    input_dict = {"input": [tensor4, tensor5], "name": "diff_shapes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List with two tensors of same shapes but different dtypes
    tensor6 = tf.constant([1, 2, 3], dtype=tf.int32)
    tensor7 = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)
    input_dict = {"input": [tensor6, tensor7], "name": "diff_dtypes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List with a single 3D tensor
    tensor8 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int64)
    input_dict = {"input": [tensor8], "name": "single_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List with multiple identical tensors
    tensor9 = tf.constant([1, 2, 3, 4], dtype=tf.int32)
    input_dict = {"input": [tensor9, tensor9], "name": "identical_tensors"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List with tensors containing negative values
    tensor10 = tf.constant([-1, 2, -3], dtype=tf.int32)
    input_dict = {"input": [tensor10], "name": "negative_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large Tensor
    tensor11 = tf.random.normal(shape=(50, 50))
    input_dict = {"input": [tensor11], "name": "large_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: List of two tensors
    tensor12 = tf.constant([[1,2],[3,4]], dtype=tf.float32)
    tensor13 = tf.constant([[5,6],[7,8]], dtype=tf.float32)
    input_dict = {"input": [tensor12, tensor13], "name": "two_tensors"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 11: List of two scalars
    tensor14 = tf.constant(2, dtype=tf.int32)
    tensor15 = tf.constant(3, dtype=tf.int32)
    input_dict = {"input": [tensor14, tensor15], "name": "two_scalars"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: List with a single boolean tensor
    tensor16 = tf.constant([True, False, True])
    input_dict = {"input": [tensor16], "name": "single_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 13: Name is none
    tensor17 = tf.constant(5.0)
    input_dict = {"input": [tensor17], "name": None}
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
