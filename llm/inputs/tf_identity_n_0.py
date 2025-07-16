
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_identity_n_inputs():
    list_of_inputs = []

    # Input 1: List of single tensor (scalar)
    input_tensor_list = [tf.constant(5)]
    input_dict = {"input": input_tensor_list, "name": "scalar_example"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of tensors (1D)
    input_tensor_list = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]
    input_dict = {"input": input_tensor_list, "name": "1d_example"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of tensors (2D)
    input_tensor_list = [tf.constant([[1, 2], [3, 4]]), tf.constant([[5, 6], [7, 8]])]
    input_dict = {"input": input_tensor_list, "name": "2d_example"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of tensors (3D)
    input_tensor_list = [tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
    input_dict = {"input": input_tensor_list, "name": "3d_example"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of tensors with different dtypes (int32 and float32)
    input_tensor_list = [tf.constant(10, dtype=tf.int32), tf.constant(3.14, dtype=tf.float32)]
    input_dict = {"input": input_tensor_list, "name": "different_dtypes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List of tensors with negative values
    input_tensor_list = [tf.constant([-1, -2, -3]), tf.constant([-4, -5, -6])]
    input_dict = {"input": input_tensor_list, "name": "negative_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List of tensors with zeros
    input_tensor_list = [tf.constant([0, 0, 0]), tf.constant([0, 0, 0])]
    input_dict = {"input": input_tensor_list, "name": "zero_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single Tensor (1D)
    input_tensor_list = [tf.constant([1, 2, 3])]
    input_dict = {"input": input_tensor_list, "name": "single_1d_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Tensor of strings (1D)
    input_tensor_list = [tf.constant(["hello", "world"])]
    input_dict = {"input": input_tensor_list, "name": "string_list"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: List of tensors with boolean values
    input_tensor_list = [tf.constant([True, False, True]), tf.constant([False, True, False])]
    input_dict = {"input": input_tensor_list, "name": "boolean_values"}
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
