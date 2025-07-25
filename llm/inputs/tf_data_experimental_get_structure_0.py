
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_get_structure_inputs():
    list_of_inputs = []

    # Input 1: A tuple containing a single 1D integer array.
    list_of_inputs.append({
        'dataset_or_iterator': (np.array([1, 2, 3], dtype=np.int32),)
    })

    # Input 2: A tuple containing a single 2D float array with negative values.
    list_of_inputs.append({
        'dataset_or_iterator': (np.array([[1.0, 2.0], [3.0, 4.0], [-5.0, -6.0]], dtype=np.float32),)
    })

    # Input 3: A tuple with two 1D arrays of different types.
    list_of_inputs.append({
        'dataset_or_iterator': (
            np.arange(10, dtype=np.int64),
            (np.random.rand(10) * 10).astype(np.uint8)
        )
    })

    # Input 4: A tuple with a string array and a 3D integer array.
    list_of_inputs.append({
        'dataset_or_iterator': (
            np.array([b'a', b'b', b'c']),
            np.array([[[1],[2]], [[3],[4]], [[5],[6]]], dtype=np.int16)
        )
    })

    # Input 5: A tuple containing a boolean array.
    list_of_inputs.append({
        'dataset_or_iterator': ((np.random.rand(5, 2) > 0.5).astype(np.bool_),)
    })

    # Input 6: A tuple with arrays of different float precisions.
    list_of_inputs.append({
        'dataset_or_iterator': (
            np.array([-10.5, -20.25], dtype=np.float16),
            np.array([[-1.5], [-3.5]], dtype=np.float64)
        )
    })

    # Input 7: A tuple with a higher-rank (4D) array.
    list_of_inputs.append({
        'dataset_or_iterator': (np.zeros((2, 2, 2, 2), dtype=np.uint16),)
    })

    # Input 8: A tuple with single-element arrays.
    list_of_inputs.append({
        'dataset_or_iterator': (
            np.array([100], dtype=np.int64),
            np.array([b'hello']),
        )
    })

    # Input 9: A tuple of empty arrays.
    list_of_inputs.append({
        'dataset_or_iterator': (
            np.array([], dtype=np.string_),
            np.array([], dtype=np.float32)
        )
    })

    # Input 10: A tuple with multiple arrays of various types and ranks.
    list_of_inputs.append({
        'dataset_or_iterator': (
            np.arange(4, dtype=np.int32),
            np.random.rand(4, 1).astype(np.float32),
            (np.random.rand(4, 2) > 0.5)
        )
    })

    # Input 11: Tuple with complex numbers.
    list_of_inputs.append({
        'dataset_or_iterator': (np.array([1+2j, -3+4j], dtype=np.complex128),)
    })

    # Input 12: Tuple with unsigned integers.
    list_of_inputs.append({
        'dataset_or_iterator': (
            np.array([0, 255], dtype=np.uint8),
            np.array([0, 65535], dtype=np.uint16),
        )
    })
    
    return list_of_inputs

generated_inputs["tf.data.experimental.get_structure"] = tf_data_experimental_get_structure_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.get_structure' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.get_structure'.")

check_valid('tf.data.experimental.get_structure', generated_inputs['tf.data.experimental.get_structure'], lib="tf", suffix=0)
