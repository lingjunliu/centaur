
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_assert_cardinality_inputs():
    list_of_inputs = []

    # Input 1: Expected cardinality is 0
    list_of_inputs.append({
        'args': {'expected_cardinality': 0},
        'inner_values': np.array([], dtype=np.float32)
    })

    # Input 2: Expected cardinality is 1
    list_of_inputs.append({
        'args': {'expected_cardinality': 1},
        'inner_values': np.array([42], dtype=np.int32)
    })

    # Input 3: Small positive cardinality
    list_of_inputs.append({
        'args': {'expected_cardinality': 10},
        'inner_values': np.arange(10, dtype=np.int64)
    })

    # Input 4: Cardinality with 2D numpy array elements
    list_of_inputs.append({
        'args': {'expected_cardinality': 5},
        'inner_values': np.random.rand(5, 3, 2).astype(np.float32)
    })

    # Input 5: Medium cardinality
    list_of_inputs.append({
        'args': {'expected_cardinality': 128},
        'inner_values': np.zeros((128, 1), dtype=np.uint8)
    })

    # Input 6: Cardinality with boolean elements
    list_of_inputs.append({
        'args': {'expected_cardinality': 4},
        'inner_values': np.array([True, False, False, True])
    })

    # Input 7: Cardinality with string elements
    list_of_inputs.append({
        'args': {'expected_cardinality': 3},
        'inner_values': np.array(['one', 'two', 'three'], dtype=object)
    })

    # Input 8: Cardinality with complex numbers
    list_of_inputs.append({
        'args': {'expected_cardinality': 2},
        'inner_values': np.array([1+1j, -2-2j], dtype=np.complex128)
    })

    # Input 9: Large cardinality
    list_of_inputs.append({
        'args': {'expected_cardinality': 500},
        'inner_values': np.linspace(0, 100, 500, dtype=np.float16)
    })

    # Input 10: Another simple case
    list_of_inputs.append({
        'args': {'expected_cardinality': 25},
        'inner_values': np.arange(25)
    })

    return list_of_inputs

generated_inputs["tf.data.experimental.assert_cardinality"] = tf_data_experimental_assert_cardinality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.assert_cardinality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.assert_cardinality'.")

check_valid('tf.data.experimental.assert_cardinality', generated_inputs['tf.data.experimental.assert_cardinality'], lib="tf", suffix=0)
