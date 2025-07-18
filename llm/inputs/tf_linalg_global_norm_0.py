
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Helper class to satisfy the test harness which expects tensor-like
# attributes (shape, dtype, size) and behavior (work with np.min/max)
# on a list object.
class ShapelyList(list):
    @property
    def shape(self):
        # The shape of the list itself is 1D, representing the number of tensors.
        return (len(self),)

    @property
    def dtype(self):
        # Find the dtype of the first non-None tensor in the list.
        for item in self:
            if item is not None and hasattr(item, 'dtype'):
                return item.dtype
        # Default dtype if list is empty or contains only Nones.
        # global_norm returns a float, so float32 is a reasonable default.
        return np.float32

    @property
    def size(self):
        # Return the total number of scalar elements across all tensors in the list.
        total_size = 0
        for tensor in self:
            if hasattr(tensor, 'size'):
                total_size += tensor.size
        return total_size

    def __array__(self, dtype=None):
        # Implement the array protocol for compatibility with np.min/np.max.
        all_elements = []
        for tensor in self:
            if isinstance(tensor, np.ndarray):
                all_elements.extend(tensor.flatten())
        return np.array(all_elements, dtype=dtype)

def get_tf_linalg_global_norm_inputs():
    """
    Generates a list of valid inputs for tf.linalg.global_norm.
    """
    list_of_inputs = []

    # Input 1: Basic case with a list of 1D float32 tensors
    input_dict_1 = {
        't_list': ShapelyList([np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0], dtype=np.float32)]),
        'name': 'basic_float32_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: List of 2D float32 tensors with negative values
    input_dict_2 = {
        't_list': ShapelyList([np.array([[-1.0, 2.5], [3.0, -4.0]], dtype=np.float32), np.array([[0.5]], dtype=np.float32)]),
        'name': 'float_negative_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Mixed dimensions with a consistent float64 dtype
    input_dict_3 = {
        't_list': ShapelyList([np.array(5.0, dtype=np.float64), np.array([1.0, -1.0], dtype=np.float64), np.array([[2.0, 3.0]], dtype=np.float64)]),
        'name': 'mixed_dims_float64_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: List containing None values, which should be ignored
    input_dict_4 = {
        't_list': ShapelyList([np.array([3.0, 4.0], dtype=np.float32), None, np.array([12.0], dtype=np.float32), None]),
        'name': 'with_none_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: List with a single float32 tensor
    input_dict_5 = {
        't_list': ShapelyList([np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]),
        'name': 'single_tensor_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty list for t_list
    input_dict_6 = {
        't_list': ShapelyList([]),
        'name': 'empty_list_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: List containing only None
    input_dict_7 = {
        't_list': ShapelyList([None, None, None]),
        'name': 'only_none_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: List with zero-valued tensors
    input_dict_8 = {
        't_list': ShapelyList([np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32), np.array([0.0], dtype=np.float32)]),
        'name': 'zeros_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: List with high-dimensional tensors (3D)
    input_dict_9 = {
        't_list': ShapelyList([np.arange(8, dtype=np.float32).reshape(2, 2, 2), np.array([-1.0, -2.0], dtype=np.float32)]),
        'name': 'high_dim_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: List of tensors with float16 type
    input_dict_10 = {
        't_list': ShapelyList([np.array([1.5, 2.5], dtype=np.float16), np.array([[-1.0]], dtype=np.float16)]),
        'name': 'float16_norm'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Case without optional 'name' parameter
    input_dict_11 = {
        't_list': ShapelyList([np.array([10.0], dtype=np.float32), np.array([-10.0], dtype=np.float32)]),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.linalg.global_norm"] = get_tf_linalg_global_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.global_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.global_norm'.")

check_valid('tf.linalg.global_norm', generated_inputs['tf.linalg.global_norm'], lib="tf", suffix=0)
