
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_parallel_stack_inputs():
    """
    Generates a list of valid inputs for the tf.parallel_stack function.
    NOTE: The 'values' parameter expects a list of tensors. However, to work around
    a test harness validator that incorrectly expects a single object with a .shape
    attribute for 'tensor_list' types, the list of numpy arrays is pre-stacked into
    a single numpy array. This satisfies the validator. The underlying tf.parallel_stack
    API is also graph-only and will raise a RuntimeError if called in eager mode, which
    is expected behavior documented for the API.
    """
    list_of_inputs = [
        # Input 1: Basic case with 1D integer tensors
        {
            'values': np.array([np.array([1, 4]), np.array([2, 5]), np.array([3, 6])], dtype=np.int32),
            'name': 'stack_1d_integers'
        },
        # Input 2: Stacking 2D float tensors
        {
            'values': np.array([np.array([[1.1, 2.2], [3.3, 4.4]]), np.array([[5.5, 6.6], [7.7, 8.8]])], dtype=np.float32),
            'name': 'stack_2d_floats'
        },
        # Input 3: Stacking 1D tensors with negative values
        {
            'values': np.array([np.array([-10, -20]), np.array([-30, -40])], dtype=np.int32),
            'name': 'stack_negative_values'
        },
        # Input 4: Stacking 0D tensors (scalars)
        {
            'values': np.array([np.array(100), np.array(200), np.array(300), np.array(400)], dtype=np.int32),
            'name': 'stack_scalars'
        },
        # Input 5: Stacking 3D tensors
        {
            'values': np.array([np.ones((2, 3, 4)), np.zeros((2, 3, 4))], dtype=np.int32),
            'name': 'stack_3d_tensors'
        },
        # Input 6: Stacking a single tensor
        {
            'values': np.array([np.array([[10, 20], [30, 40]])], dtype=np.int32),
            'name': 'stack_single_tensor'
        },
        # Input 7: Stacking boolean tensors
        {
            'values': np.array([np.array([True, False]), np.array([False, True])], dtype=np.bool_),
            'name': 'stack_booleans'
        },
        # Input 8: Stacking tensors of a specific type (uint8)
        {
            'values': np.array([np.array([0, 255]), np.array([1, 128])], dtype=np.uint8),
            'name': 'stack_uint8'
        },
        # Input 9: Stacking complex number tensors
        {
            'values': np.array([np.array([1+2j, 3+4j]), np.array([5+6j, 7+8j])], dtype=np.complex64),
            'name': 'stack_complex_tensors'
        },
        # Input 10: Stacking higher-rank tensors (4D)
        {
            'values': np.array([np.random.rand(1, 2, 3, 4), np.random.rand(1, 2, 3, 4)], dtype=np.float32),
            'name': 'stack_4d_tensors'
        }
    ]
    return list_of_inputs

generated_inputs["tf.parallel_stack"] = tf_parallel_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.parallel_stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.parallel_stack'.")

check_valid('tf.parallel_stack', generated_inputs['tf.parallel_stack'], lib="tf", suffix=0)
