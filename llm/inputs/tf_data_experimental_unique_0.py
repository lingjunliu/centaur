
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_unique_inputs():
    list_of_inputs = []

    # The API `tf.data.experimental.unique()` returns a transformation function
    # that is meant to be used with `tf.data.Dataset.apply()`. The test harness
    # needs to create the initial dataset. The recurring error
    # "returns a function, but the input does not have inner values" suggests that
    # the harness expects the data for the dataset under a specific key.
    # Previous attempts with other keys failed. This attempt uses the key 'dataset'.
    # The harness is expected to use the value of 'dataset' to create the
    # `tf.data.Dataset` object, and then call the `unique()` function with an
    # empty dictionary of arguments, which conforms to its `{}` signature.

    # Input 1: Basic case with integers and duplicates
    list_of_inputs.append({'dataset': np.array([1, 37, 2, 37, 2, 1], dtype=np.int32)})

    # Input 2: Floating point numbers with duplicates
    list_of_inputs.append({'dataset': np.array([1.1, 2.2, 1.1, 3.3, 2.2, 4.4], dtype=np.float32)})

    # Input 3: Strings with duplicates
    list_of_inputs.append({'dataset': np.array(["apple", "banana", "apple", "cherry", "banana"], dtype=object)})

    # Input 4: Negative integers and zero
    list_of_inputs.append({'dataset': np.array([-1, 0, 2, -1, 0, -3], dtype=np.int64)})

    # Input 5: All elements are the same
    list_of_inputs.append({'dataset': np.array([5, 5, 5, 5, 5], dtype=np.int32)})

    # Input 6: All elements are unique
    list_of_inputs.append({'dataset': np.array([10, 20, 30, 40, 50], dtype=np.int32)})

    # Input 7: Empty dataset
    list_of_inputs.append({'dataset': np.array([], dtype=np.float64)})

    # Input 8: Dataset with a single element
    list_of_inputs.append({'dataset': np.array([100], dtype=np.int32)})

    # Input 9: Dataset with 2D elements (vectors)
    list_of_inputs.append({'dataset': np.array([[1, 2], [3, 4], [1, 2], [5, 6]], dtype=np.int32)})

    # Input 10: Dataset with 3D elements
    list_of_inputs.append({'dataset': np.array([[[1],[2]], [[3],[4]], [[1],[2]]], dtype=np.int32)})

    return list_of_inputs

generated_inputs["tf.data.experimental.unique"] = tf_data_experimental_unique_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.unique'.")

check_valid('tf.data.experimental.unique', generated_inputs['tf.data.experimental.unique'], lib="tf", suffix=0)
