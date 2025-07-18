
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_clip_by_global_norm_inputs():
    """
    Generates a list of valid inputs for tf.clip_by_global_norm.
    """
    list_of_inputs = []

    # Input 1: Basic case where clipping occurs
    input_dict_1 = {
        't_list': [np.array([3.0, 4.0], dtype=np.float32), np.array([5.0, 12.0], dtype=np.float32)],
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': None,
        'name': 'basic_clipping'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: No clipping occurs
    input_dict_2 = {
        't_list': [np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)],
        'clip_norm': np.array(10.0, dtype=np.float32),
        'use_norm': None,
        'name': 'no_clipping'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Mixed shapes and negative values
    input_dict_3 = {
        't_list': [np.array([-1.0, 2.0], dtype=np.float32), np.array([[-2.0], [3.0], [-4.0]], dtype=np.float32)],
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': None,
        'name': 'mixed_shapes'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using `use_norm` to specify the global norm
    input_dict_4 = {
        't_list': [np.array([6.0, 8.0], dtype=np.float64)],
        'clip_norm': np.array(5.0, dtype=np.float64),
        'use_norm': np.array(12.0, dtype=np.float64),
        'name': 'with_use_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: global_norm is exactly equal to clip_norm
    input_dict_5 = {
        't_list': [np.array([3.0, 4.0], dtype=np.float32)],
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': None,
        'name': 'norm_equals_clip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Tensors with all zeros
    input_dict_6 = {
        't_list': [np.zeros((2, 2), dtype=np.float32), np.zeros((3, 1), dtype=np.float32)],
        'clip_norm': np.array(1.0, dtype=np.float32),
        'use_norm': None,
        'name': 'zero_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: High-dimensional tensor
    input_dict_7 = {
        't_list': [np.ones((2, 2, 2), dtype=np.float32)],
        'clip_norm': np.array(1.0, dtype=np.float32),
        'use_norm': None,
        'name': 'high_dim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Using integer types
    input_dict_8 = {
        't_list': [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)],
        'clip_norm': np.array(2.0, dtype=np.float32),
        'use_norm': None,
        'name': 'integer_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using float64 for higher precision
    input_dict_9 = {
        't_list': [np.array([1e10, 2e10], dtype=np.float64), np.array([3e10, 4e10], dtype=np.float64)],
        'clip_norm': np.array(1e10, dtype=np.float64),
        'use_norm': None,
        'name': 'float64_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Empty list of tensors
    input_dict_10 = {
        't_list': [],
        'clip_norm': np.array(1.0, dtype=np.float32),
        'use_norm': None,
        'name': 'empty_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.clip_by_global_norm"] = tf_clip_by_global_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.clip_by_global_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.clip_by_global_norm'.")

check_valid('tf.clip_by_global_norm', generated_inputs['tf.clip_by_global_norm'], lib="tf", suffix=0)
