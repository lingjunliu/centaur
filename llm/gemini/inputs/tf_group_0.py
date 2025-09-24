
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_group_inputs():
    list_of_inputs = []

    # Input 1: Two 1D integer tensors, stacked.
    input_dict_1 = {
        'inputs': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        'name': 'group_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Three 2D float tensors, stacked.
    input_dict_2 = {
        'inputs': np.array([[[1.1, 2.2]], [[3.3, 4.4]], [[5.5, 6.6]]], dtype=np.float32),
        'name': 'float_group'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: A single tensor, stacked (adds a dimension).
    input_dict_3 = {
        'inputs': np.array([[[1], [2], [3]]], dtype=np.int16),
        'name': 'single_tensor_group'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: An empty list of tensors, represented by an empty numpy array.
    input_dict_4 = {
        'inputs': np.array([], dtype=np.float64),
        'name': 'empty_group'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tensors with negative values, stacked.
    input_dict_5 = {
        'inputs': np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32),
        'name': 'negative_group'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Scalar tensors, stacked into a 1D array.
    input_dict_6 = {
        'inputs': np.array([10, 20, 30, 40], dtype=np.int64),
        'name': 'scalar_group'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Boolean tensors, stacked.
    input_dict_7 = {
        'inputs': np.array([[True, False], [True, True], [False, False]]),
        'name': 'boolean_group'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Complex value tensors, stacked.
    input_dict_8 = {
        'inputs': np.array([[1+2j, 5-6j], [3+4j, 7-8j]], dtype=np.complex64),
        'name': 'complex_group'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: High-dimensional tensors, stacked.
    input_dict_9 = {
        'inputs': np.stack([np.arange(12).reshape(2, 3, 2).astype(np.int32),
                           np.ones((2, 3, 2), dtype=np.int32) * -1]),
        'name': 'high_dim_group'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensors including zeros, stacked.
    input_dict_10 = {
        'inputs': np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, -1.0]]], dtype=np.float32),
        'name': 'group_with_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["tf.group"] = tf_group_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.group' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.group'.")

check_valid('tf.group', generated_inputs['tf.group'], lib="tf", suffix=0)
