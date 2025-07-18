
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_data_experimental_take_while_inputs():
    list_of_inputs = []

    # The test harness requires the dataset object but fails on deepcopy.
    # The "no inner values" error suggests it can't find the dataset.
    # We hypothesize the harness expects the dataset's source data under the key 'self'
    # and will construct the Dataset object internally.
    # The 'predicate' argument must be a list as per the strict signature.

    # Input 1: Basic integer array
    input_dict = {
        'self': np.arange(10, dtype=np.int32),
        'predicate': [True, True, True, False, True]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Floating point array
    input_dict = {
        'self': np.linspace(0., 1., 10, dtype=np.float32),
        'predicate': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative numbers array
    input_dict = {
        'self': np.arange(-5, 5, dtype=np.int64),
        'predicate': [1, 0, -1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array (vector elements)
    input_dict = {
        'self': np.arange(20, dtype=np.int32).reshape(10, 2),
        'predicate': [1.1, 2.2, 3.3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean array
    input_dict = {
        'self': np.array([True, True, False, True]),
        'predicate': [True, 0, 'a', None]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty array
    input_dict = {
        'self': np.array([], dtype=np.float64),
        'predicate': [True, False]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tuple of arrays for structured elements
    input_dict = {
        'self': (np.arange(10, dtype=np.int32), -np.arange(10, dtype=np.int32)),
        'predicate': list(range(50))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array elements
    input_dict = {
        'self': np.random.rand(5, 2, 2).astype(np.float32),
        'predicate': [[1], [2, 3], [4, 5, 6]]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Dictionary of arrays for structured elements
    input_dict = {
        'self': {'a': np.arange(5, dtype=np.int32), 'b': np.array([True, True, True, False, True])},
        'predicate': ['predicate', 'list']
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single element array
    input_dict = {
        'self': np.array([100], dtype=np.int32),
        'predicate': [True]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.take_while"] = tf_data_experimental_take_while_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.take_while' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.take_while'.")

check_valid('tf.data.experimental.take_while', generated_inputs['tf.data.experimental.take_while'], lib="tf", suffix=0)
