
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_clip_by_global_norm_inputs():
    """
    Generates a list of valid inputs for tf.clip_by_global_norm.
    """
    list_of_inputs = []

    # Input 1: Basic clipping case. Tensors in list have same shape.
    input_dict_1 = {
        't_list': np.array([np.array([3.0, 4.0], dtype=np.float32), np.array([5.0, 1.0], dtype=np.float32)]),
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': None,
        'name': 'basic_clipping'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: No clipping. Tensors in list have same shape.
    input_dict_2 = {
        't_list': np.array([np.array([1.0, 2.0], dtype=np.float32), np.array([-1.0, 1.0], dtype=np.float32)]),
        'clip_norm': np.array(4.0, dtype=np.float32),
        'use_norm': None,
        'name': 'no_clipping'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Tensors with higher rank, but same shape.
    input_dict_3 = {
        't_list': np.array([
            np.array([[-3.0, 4.0], [1.0, 2.0]], dtype=np.float32),
            np.array([[1.0, 5.0], [-2.0, -3.0]], dtype=np.float32)
        ]),
        'clip_norm': np.array(7.0, dtype=np.float32),
        'use_norm': None,
        'name': 'higher_rank_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using the `use_norm` parameter. List with one tensor.
    input_dict_4 = {
        't_list': np.array([np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)]),
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': np.array(10.0, dtype=np.float32),
        'name': 'with_use_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tensors containing all zeros with the same shape.
    input_dict_5 = {
        't_list': np.array([np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]),
        'clip_norm': np.array(1.0, dtype=np.float32),
        'use_norm': None,
        'name': 'zero_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Single tensor in the list.
    input_dict_6 = {
        't_list': np.array([np.array([5.0, 12.0], dtype=np.float32)]),
        'clip_norm': np.array(10.0, dtype=np.float32),
        'use_norm': None,
        'name': 'single_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Using float64 data type.
    input_dict_7 = {
        't_list': np.array([np.array([0.5, 0.6], dtype=np.float64), np.array([0.7, 0.1], dtype=np.float64)]),
        'clip_norm': np.array(1.0, dtype=np.float64),
        'use_norm': None,
        'name': 'float64_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: global_norm is exactly equal to clip_norm.
    input_dict_8 = {
        't_list': np.array([np.array([6.0, -8.0], dtype=np.float32)]),
        'clip_norm': np.array(10.0, dtype=np.float32),
        'use_norm': None,
        'name': 'norm_equals_clip_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using `use_norm` that is smaller than `clip_norm`.
    input_dict_9 = {
        't_list': np.array([np.array([100.0, 200.0], dtype=np.float32)]),
        'clip_norm': np.array(50.0, dtype=np.float32),
        'use_norm': np.array(20.0, dtype=np.float32),
        'name': 'use_norm_less_than_clip_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Large values leading to large norm.
    input_dict_10 = {
        't_list': np.array([np.array([1e6, -2e6], dtype=np.float32)]),
        'clip_norm': np.array(1e5, dtype=np.float32),
        'use_norm': None,
        'name': 'large_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Empty list of tensors
    input_dict_11 = {
        't_list': np.array([], dtype=np.float32),
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': None,
        'name': 'empty_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))


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
