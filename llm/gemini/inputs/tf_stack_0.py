
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_stack_inputs():
    """
    Generates a list of valid inputs for the tf.stack function.
    """
    list_of_inputs = []

    # Input 1: Basic case with 1D tensors, default axis=0
    input_dict = {
        'values': np.array([[1, 4], [2, 5], [3, 6]]),
        'axis': 0,
        'name': 'stack_1d_axis_0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Stacking 1D tensors along axis=1
    input_dict = {
        'values': np.array([[1, 4], [2, 5], [3, 6]]),
        'axis': 1,
        'name': 'stack_1d_axis_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Stacking 2D tensors, axis=0
    input_dict = {
        'values': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        'axis': 0,
        'name': 'stack_2d_axis_0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Stacking 2D tensors, axis=1
    input_dict = {
        'values': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        'axis': 1,
        'name': 'stack_2d_axis_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Stacking 2D tensors, axis=2 (the last possible dimension)
    input_dict = {
        'values': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        'axis': 2,
        'name': 'stack_2d_axis_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Using a negative axis (-1) for 2D tensors
    input_dict = {
        'values': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        'axis': -1,
        'name': 'stack_2d_neg_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Stacking 3D tensors with float32 dtype
    input_dict = {
        'values': np.array([
            np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            np.arange(8, 16, dtype=np.float32).reshape(2, 2, 2)
        ]),
        'axis': 1,
        'name': 'stack_3d_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Stacking scalars (rank-0 tensors)
    input_dict = {
        'values': np.array([10, 20, 30, 40]),
        'axis': 0,
        'name': 'stack_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Stacking boolean tensors
    input_dict = {
        'values': np.array([[True, False], [False, True]]),
        'axis': 1,
        'name': 'stack_bools'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Using another negative axis (-2) for 1D tensors
    input_dict = {
        'values': np.array([[1, 4], [2, 5], [3, 6]]),
        'axis': -2,
        'name': 'stack_1d_neg_axis_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Stacking a larger number of tensors
    input_dict = {
        'values': np.array([np.ones((2, 2), dtype=np.int32) * i for i in range(10)]),
        'axis': 0,
        'name': 'stack_long_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Stacking tensors with an empty dimension
    input_dict = {
        'values': np.array([np.zeros((3, 0, 4)), np.zeros((3, 0, 4)), np.zeros((3, 0, 4))]),
        'axis': 1,
        'name': 'stack_empty_dim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.stack"] = tf_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.stack'.")

check_valid('tf.stack', generated_inputs['tf.stack'], lib="tf", suffix=0)
