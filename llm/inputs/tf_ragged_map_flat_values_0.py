
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_map_flat_values_inputs():
    """
    Generates a list of valid inputs for tf.ragged.map_flat_values.
    The 'op' parameter must be a callable. The user's provided signature
    {'op': 'tensor'} is incorrect for this API and causes a TypeError at runtime.
    This implementation provides the correct callable type for 'op' to create
    a valid input for the TensorFlow API, even though it may conflict with the
    user's validation framework.
    """
    list_of_inputs = []

    # Input 1: Simple case with tf.negative
    list_of_inputs.append({
        'op': tf.negative,
        '*args': [tf.ragged.constant([[1, 2, 3], [], [4, 5]])]
    })

    # Input 2: Using tf.abs with negative and zero values
    list_of_inputs.append({
        'op': tf.abs,
        '*args': [tf.ragged.constant([[-1, 0, -5], [], [4, -100]], dtype=tf.float32)]
    })

    # Input 3: Floating point values with tf.square
    list_of_inputs.append({
        'op': tf.square,
        '*args': [tf.ragged.constant([[1.5, -2.5], [3.0], []])]
    })

    # Input 4: Two ragged tensor arguments with tf.add
    list_of_inputs.append({
        'op': tf.add,
        '*args': [
            tf.ragged.constant([[1, 2], [3], [4, 5, 6]]),
            tf.ragged.constant([[10, 20], [30], [40, 50, 60]])
        ]
    })

    # Input 5: Ragged tensor and a scalar tensor argument
    list_of_inputs.append({
        'op': tf.multiply,
        '*args': [
            tf.ragged.constant([[1, -2], [30]]),
            tf.constant(10, dtype=tf.int32)
        ]
    })

    # Input 6: Higher ragged_rank (rank 2)
    list_of_inputs.append({
        'op': tf.identity,
        '*args': [
            tf.ragged.constant([[[1, 2], [3]], [[], [4, 5, 6]], []])
        ]
    })

    # Input 7: Two ragged tensors with rank 2 and tf.maximum
    list_of_inputs.append({
        'op': tf.maximum,
        '*args': [
            tf.ragged.constant([[[1, -5], [3]], [[], [40, -5, 6]]]),
            tf.ragged.constant([[[0, 5], [-3]], [[], [4, 50, -6]]])
        ]
    })

    # Input 8: Empty ragged tensor
    list_of_inputs.append({
        'op': tf.square,
        '*args': [
            tf.ragged.constant([], dtype=tf.int32)
        ]
    })

    # Input 9: Ragged tensor with only empty rows
    list_of_inputs.append({
        'op': tf.negative,
        '*args': [
            tf.ragged.constant([[], [], []])
        ]
    })

    # Input 10: Boolean ragged tensor with tf.logical_not
    list_of_inputs.append({
        'op': tf.math.logical_not,
        '*args': [
            tf.ragged.constant([[True, False], [True], []])
        ]
    })

    # Input 11: String ragged tensor with tf.strings.length
    list_of_inputs.append({
        'op': tf.strings.length,
        '*args': [
            tf.ragged.constant([['hello', 'world'], ['tensorflow', ''], []], dtype=tf.string)
        ]
    })

    # Input 12: Ragged tensor with float64 values and tf.math.sqrt
    list_of_inputs.append({
        'op': tf.math.sqrt,
        '*args': [
            tf.ragged.constant([[1e8, 4e8], [9e8, 16e8]], dtype=tf.float64)
        ]
    })

    return [copy.deepcopy(i) for i in list_of_inputs]

generated_inputs["tf.ragged.map_flat_values"] = get_map_flat_values_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ragged.map_flat_values' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.map_flat_values'.")

check_valid('tf.ragged.map_flat_values', generated_inputs['tf.ragged.map_flat_values'], lib="tf", suffix=0)
