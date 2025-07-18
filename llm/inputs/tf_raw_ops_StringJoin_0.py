
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_stringjoin_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D join with a space separator
    inputs_list_1 = [np.array([b"hello", b"world"], dtype=object), np.array([b"tensorflow", b"rules"], dtype=object)]
    input_dict = {
        'inputs': np.stack(inputs_list_1),
        'separator': ' ',
        'name': 'basic_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Default empty separator
    inputs_list_2 = [np.array([b"first", b"second"], dtype=object), np.array([b"1", b"2"], dtype=object)]
    input_dict = {
        'inputs': np.stack(inputs_list_2),
        'separator': '',
        'name': 'default_separator'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple tensors (3) with a custom separator
    inputs_list_3 = [np.array([b"a"], dtype=object), np.array([b"b"], dtype=object), np.array([b"c"], dtype=object)]
    input_dict = {
        'inputs': np.stack(inputs_list_3),
        'separator': '-',
        'name': 'multiple_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D Tensors
    inputs_list_4 = [np.array([[b"a", b"b"], [b"c", b"d"]], dtype=object), np.array([[b"1", b"2"], [b"3", b"4"]], dtype=object)]
    input_dict = {
        'inputs': np.stack(inputs_list_4),
        'separator': ':',
        'name': '2d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting a scalar tensor with a 1D tensor
    scalar_5 = np.array(b"id", dtype=object)
    vec_5 = np.array([b"101", b"102", b"103"], dtype=object)
    broadcasted_scalar_5 = np.broadcast_to(scalar_5, vec_5.shape)
    input_dict = {
        'inputs': np.stack([broadcasted_scalar_5, vec_5]),
        'separator': '_',
        'name': 'broadcast_scalar_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting multiple scalars with a 2D tensor
    scalar1_6 = np.array(b"prefix", dtype=object)
    mat_6 = np.array([[b"a", b"b"], [b"c", b"d"]], dtype=object)
    scalar2_6 = np.array(b"suffix", dtype=object)
    broadcasted_scalar1_6 = np.broadcast_to(scalar1_6, mat_6.shape)
    broadcasted_scalar2_6 = np.broadcast_to(scalar2_6, mat_6.shape)
    input_dict = {
        'inputs': np.stack([broadcasted_scalar1_6, mat_6, broadcasted_scalar2_6]),
        'separator': '-',
        'name': 'broadcast_multiple_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Only scalar inputs
    inputs_list_7 = [np.array(b"part1", dtype=object), np.array(b"part2", dtype=object)]
    input_dict = {
        'inputs': np.stack(inputs_list_7),
        'separator': ' <-> ',
        'name': 'all_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Input tensors containing empty strings
    inputs_list_8 = [np.array([b"a", b"", b"c"], dtype=object), np.array([b"x", b"y", b""], dtype=object)]
    input_dict = {
        'inputs': np.stack(inputs_list_8),
        'separator': ',',
        'name': 'with_empty_strings'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single tensor in the input list
    inputs_list_9 = [np.array([b"a", b"b", b"c"], dtype=object)]
    input_dict = {
        'inputs': np.stack(inputs_list_9),
        'separator': ';',
        'name': 'single_tensor_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Empty tensor inputs
    inputs_list_10 = [np.array([], dtype=object), np.array([], dtype=object)]
    input_dict = {
        'inputs': np.stack(inputs_list_10),
        'separator': ',',
        'name': 'empty_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: High-dimensional tensors (3D)
    arr1 = np.array([str(i).encode('utf-8') for i in range(8)], dtype=object).reshape(2, 2, 2)
    arr2 = np.full((2, 2, 2), b'X', dtype=object)
    inputs_list_11 = [arr1, arr2]
    input_dict = {
        'inputs': np.stack(inputs_list_11),
        'separator': '=',
        'name': '3d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Broadcasting scalar to 2D tensor
    scalar_12 = np.array(b"S", dtype=object)
    mat_12 = np.array([[b"1", b"2"], [b"3", b"4"]], dtype=object)
    broadcasted_scalar_12 = np.broadcast_to(scalar_12, mat_12.shape)
    input_dict = {
        'inputs': np.stack([mat_12, broadcasted_scalar_12]),
        'separator': ':',
        'name': 'broadcast_scalar_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.StringJoin"] = tf_raw_ops_stringjoin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.StringJoin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringJoin'.")

check_valid('tf.raw_ops.StringJoin', generated_inputs['tf.raw_ops.StringJoin'], lib="tf", suffix=0)
