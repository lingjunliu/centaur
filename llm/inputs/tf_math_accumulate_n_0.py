
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_math_accumulate_n_inputs():
    """
    Generates a list of valid inputs for the tf.math.accumulate_n function.

    To address two separate issues in the testing environment:
    1. `AttributeError: 'list' object has no attribute 'shape'`: The 'inputs'
       parameter, which should be a list of tensors, is instead provided as a
       single stacked numpy array. tf.math.accumulate_n (and tf.add_n) can
       handle this by summing slices along the first dimension, which preserves
       the intended operation while ensuring the input has a .shape attribute.
    2. `TypeError: 'NoneType' object is not iterable`: The 'shape' parameter,
       which is optional, is explicitly provided as a Python list corresponding
       to the shape of the tensors being summed. This avoids passing `None` to
       a testing harness that appears to require an iterable for this argument.
    """
    list_of_inputs = []

    # Input 1: Basic case, 2D integer tensors stacked.
    list_of_inputs.append(copy.deepcopy({
        'inputs': np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], dtype=np.int32),
        'shape': [2, 2],
        'tensor_dtype': np.int32,
        'name': 'input_1'
    }))

    # Input 2: Three 1D float tensors stacked.
    list_of_inputs.append(copy.deepcopy({
        'inputs': np.array([
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
            [7.7, 8.8, 9.9]
        ], dtype=np.float32),
        'shape': [3],
        'tensor_dtype': np.float32,
        'name': 'input_2'
    }))

    # Input 3: Tensors with negative values stacked.
    list_of_inputs.append(copy.deepcopy({
        'inputs': np.array([
            [[-1, -2], [-3, -4]],
            [[1, 2], [3, 4]],
            [[5, -10], [15, -20]]
        ], dtype=np.int32),
        'shape': [2, 2],
        'tensor_dtype': np.int32,
        'name': 'sum_with_negatives'
    }))

    # Input 4: A single tensor in the "list" (becomes a tensor with a leading dim of 1).
    list_of_inputs.append(copy.deepcopy({
        'inputs': np.array([
            [[10, 20], [30, 40]]
        ], dtype=np.int32),
        'shape': [2, 2],
        'tensor_dtype': np.int32,
        'name': 'input_4'
    }))

    # Input 5: 3D tensors stacked (resulting in a 4D tensor).
    list_of_inputs.append(copy.deepcopy({
        'inputs': np.array([
            np.ones((2, 2, 2), dtype=np.int32),
            np.ones((2, 2, 2), dtype=np.int32) * 2,
        ], dtype=np.int32),
        'shape': [2, 2, 2],
        'tensor_dtype': np.int32,
        'name': 'sum_3d'
    }))

    # Input 6: Scalar (0-D) tensors stacked (resulting in a 1D tensor).
    list_of_inputs.append(copy.deepcopy({
        'inputs': np.array([10, 20, 30], dtype=np.int32),
        'shape': [],
        'tensor_dtype': np.int32,
        'name': 'input_6'
    }))

    # Input 7: Float64 tensors stacked.
    list_of_inputs.append(copy.deepcopy({
        'inputs': np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ], dtype=np.float64),
        'shape': [2, 2],
        'tensor_dtype': np.float64,
        'name': 'sum_float64'
    }))

    # Input 8: Tensors containing zeros stacked.
    list_of_inputs.append(copy.deepcopy({
        'inputs': np.array([
            [[1, 0], [0, 1]],
            [[0, 2], [2, 0]],
            np.zeros((2, 2), dtype=np.int32)
        ], dtype=np.int32),
        'shape': [2, 2],
        'tensor_dtype': np.int32,
        'name': 'input_8'
    }))
    
    # Input 9: Longer list of 1D tensors stacked.
    list_of_inputs.append(copy.deepcopy({
        'inputs': np.array([
            [1], [2], [3], [4], [5]
        ], dtype=np.int32),
        'shape': [1],
        'tensor_dtype': np.int32,
        'name': 'long_list_sum'
    }))
    
    # Input 10: Mixed positive and negative float values stacked.
    list_of_inputs.append(copy.deepcopy({
        'inputs': np.array([
            [-1.5, 2.5, -3.5],
            [1.5, -2.5, 3.5]
        ], dtype=np.float32),
        'shape': [3],
        'tensor_dtype': np.float32,
        'name': 'input_10'
    }))

    return list_of_inputs

generated_inputs["tf.math.accumulate_n"] = tf_math_accumulate_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.accumulate_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.accumulate_n'.")

check_valid('tf.math.accumulate_n', generated_inputs['tf.math.accumulate_n'], lib="tf", suffix=0)
