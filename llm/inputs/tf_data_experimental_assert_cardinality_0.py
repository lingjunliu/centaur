
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_assert_cardinality_inputs():
    list_of_inputs = []

    # Input 1: Positive integer
    expected_cardinality = 10
    input_dict = {"expected_cardinality": np.int32(expected_cardinality)}
    list_of_inputs.append(input_dict)

    # Input 2: Zero
    expected_cardinality = 0
    input_dict = {"expected_cardinality": np.int32(expected_cardinality)}
    list_of_inputs.append(input_dict)

    # Input 3: Large positive integer
    expected_cardinality = 1000000
    input_dict = {"expected_cardinality": np.int64(expected_cardinality)}
    list_of_inputs.append(input_dict)

    # Input 4: Another positive integer
    expected_cardinality = 42
    input_dict = {"expected_cardinality": np.int32(expected_cardinality)}
    list_of_inputs.append(input_dict)

    # Input 5: Small positive integer
    expected_cardinality = 1
    input_dict = {"expected_cardinality": np.int32(expected_cardinality)}
    list_of_inputs.append(input_dict)

    # Input 6: Prime Number
    expected_cardinality = 73
    input_dict = {"expected_cardinality": np.int32(expected_cardinality)}
    list_of_inputs.append(input_dict)
    
    # Input 7: Power of 2
    expected_cardinality = 64
    input_dict = {"expected_cardinality": np.int32(expected_cardinality)}
    list_of_inputs.append(input_dict)

    # Input 8: A different prime
    expected_cardinality = 101
    input_dict = {"expected_cardinality": np.int32(expected_cardinality)}
    list_of_inputs.append(input_dict)

    # Input 9: another positive integer
    expected_cardinality = 1234
    input_dict = {"expected_cardinality": np.int32(expected_cardinality)}
    list_of_inputs.append(input_dict)

    # Input 10: Another different integer
    expected_cardinality = 56789
    input_dict = {"expected_cardinality": np.int32(expected_cardinality)}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.assert_cardinality"] = tf_data_experimental_assert_cardinality_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.assert_cardinality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.assert_cardinality'.")

check_valid('tf.data.experimental.assert_cardinality', generated_inputs['tf.data.experimental.assert_cardinality'], lib="tf", suffix=0)
