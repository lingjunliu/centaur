
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_identity_n_inputs():
    list_of_inputs = []

    # Input 1: List of empty tensors
    input_tensor_list = [tf.constant([], dtype=tf.float32)]
    input_dict = {"input": input_tensor_list, "name": "empty_tensors"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of single float tensor
    input_tensor_list = [tf.constant(1.0, dtype=tf.float32)]
    input_dict = {"input": input_tensor_list, "name": "single_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of single int tensor
    input_tensor_list = [tf.constant(5, dtype=tf.int32)]
    input_dict = {"input": input_tensor_list, "name": "single_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of string tensor
    input_tensor_list = [tf.constant("hello")]
    input_dict = {"input": input_tensor_list, "name": "single_string"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of two tensors (int and float)
    input_tensor_list = [tf.constant(10, dtype=tf.int32), tf.constant(2.5, dtype=tf.float32)]
    input_dict = {"input": input_tensor_list, "name": "two_tensors"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List of two tensors (different shapes)
    input_tensor_list = [tf.constant([1, 2, 3], dtype=tf.int32), tf.constant([[4, 5], [6, 7]], dtype=tf.int32)]
    input_dict = {"input": input_tensor_list, "name": "different_shapes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List of 1D tensor
    input_tensor_list = [tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)]
    input_dict = {"input": input_tensor_list, "name": "1d_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of 2D tensor
    input_tensor_list = [tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.int32)]
    input_dict = {"input": input_tensor_list, "name": "2d_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List of 3D tensor
    input_tensor_list = [tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int32)]
    input_dict = {"input": input_tensor_list, "name": "3d_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: List of tensors with negative values
    input_tensor_list = [tf.constant([-1, -2, -3], dtype=tf.int32), tf.constant([-1.5, 2.5], dtype=tf.float32)]
    input_dict = {"input": input_tensor_list, "name": "negative_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: List of single bool tensor
    input_tensor_list = [tf.constant(True, dtype=tf.bool)]
    input_dict = {"input": input_tensor_list, "name": "single_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: List of tensors with different dtypes, remove string as it causes issues.
    input_tensor_list = [tf.constant(1, dtype=tf.int32), tf.constant(2.0, dtype=tf.float64), tf.constant(False, dtype=tf.bool)]
    input_dict = {"input": input_tensor_list, "name": "different_dtypes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: List of scalars
    input_tensor_list = [tf.constant(1), tf.constant(2.0), tf.constant(3, dtype=tf.int64)]
    input_dict = {"input": input_tensor_list, "name": "scalars"}
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
