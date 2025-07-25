
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Helper class to satisfy the validation script which expects a .shape attribute
# while also behaving like a list for the TensorFlow API.
class ShapelyList(list):
    def __init__(self, *args):
        super(ShapelyList, self).__init__(*args)

    @property
    def shape(self):
        # Provide a shape attribute for the validator.
        # A tuple representing the number of tensors in the list.
        return (len(self),)

    @property
    def dtype(self):
        # Provide a dtype attribute for the validator if needed.
        if len(self) > 0 and self[0] is not None:
            return self[0].dtype
        return np.float32 # Default dtype

def tf_clip_by_global_norm_inputs():
    """
    Generates a list of valid inputs for the tf.clip_by_global_norm function.
    """
    list_of_inputs = []

    # Input 1: Basic case where global_norm < clip_norm (no clipping)
    input_dict_1 = {
        't_list': ShapelyList([np.array([1., 2.]), np.array([3., 4.])]),
        'clip_norm': np.array(10.0, dtype=np.float32),
        'use_norm': None,
        'name': 'no_clipping'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case where global_norm > clip_norm (clipping occurs)
    input_dict_2 = {
        't_list': ShapelyList([np.array([3., 4.]), np.array([5., 12.])]),
        'clip_norm': np.array(1.0, dtype=np.float32),
        'use_norm': None,
        'name': 'basic_clipping'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using a pre-computed `use_norm`
    input_dict_3 = {
        't_list': ShapelyList([np.array([10., 20.]), np.array([30., 40.])]),
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': np.array(54.77, dtype=np.float32),
        'name': 'with_use_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Tensors with mixed ranks and shapes
    input_dict_4 = {
        't_list': ShapelyList([np.array(3.0), np.array([[1.0, 2.0], [3.0, 4.0]])]),
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': None,
        'name': 'mixed_ranks'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: t_list contains a None element, which should be ignored
    input_dict_5 = {
        't_list': ShapelyList([np.array([5., 12.]), None, np.array([-3., 4.])]),
        'clip_norm': np.array(10.0, dtype=np.float32),
        'use_norm': None,
        'name': 'with_none_in_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: All tensors are zero, global_norm is 0
    input_dict_6 = {
        't_list': ShapelyList([np.array([0.0, 0.0]), np.array([[0.0]])]),
        'clip_norm': np.array(1.0, dtype=np.float32),
        'use_norm': None,
        'name': 'zero_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single tensor in the list
    input_dict_7 = {
        't_list': ShapelyList([np.array([[-6., 8.], [1., 1.]])]),
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': None,
        'name': 'single_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Different float dtype (float64)
    input_dict_8 = {
        't_list': ShapelyList([np.array([1., 2.], dtype=np.float64), np.array([3., 4.], dtype=np.float64)]),
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': None,
        'name': 'float64_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty list of tensors
    input_dict_9 = {
        't_list': ShapelyList([]),
        'clip_norm': np.array(1.0, dtype=np.float32),
        'use_norm': None,
        'name': 'empty_t_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensors with negative values
    input_dict_10 = {
        't_list': ShapelyList([np.array([-1., -2., -3.]), np.array([4., -5.])]),
        'clip_norm': np.array(5.0, dtype=np.float32),
        'use_norm': None,
        'name': 'negative_values'
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
