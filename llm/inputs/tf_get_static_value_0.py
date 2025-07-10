
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_get_static_value_inputs():
    list_of_inputs = []

    # Input 1: Simple constant tensor, partial=False
    tensor = tf.constant(10)
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple constant tensor, partial=True
    tensor = tf.constant(20)
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D constant tensor, partial=False
    tensor = tf.constant([1, 2, 3])
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D constant tensor, partial=True
    tensor = tf.constant([[4, 5], [6, 7]])
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D constant tensor, partial=False
    tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with negative values, partial=True
    tensor = tf.constant([-1, -2, -3])
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with zeros, partial=False
    tensor = tf.constant([0, 0, 0])
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with mixed positive and negative values, partial=True
    tensor = tf.constant([-1, 2, -3, 4])
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with different data type, partial=False
    tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with shape (0,), partial=True
    tensor = tf.constant([], dtype=tf.int32)
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.get_static_value"] = tf_get_static_value_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.get_static_value' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.get_static_value'.")

check_valid('tf.get_static_value', generated_inputs['tf.get_static_value'], lib="tf", suffix=0)
