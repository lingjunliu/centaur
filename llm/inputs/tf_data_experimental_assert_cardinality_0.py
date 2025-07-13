
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_assert_cardinality_inputs():
    list_of_inputs = []

    # Input 1: Positive integer
    input_dict = {"expected_cardinality": int(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    input_dict = {"expected_cardinality": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Large positive integer
    input_dict = {"expected_cardinality": int(100000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: One
    input_dict = {"expected_cardinality": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Another positive integer
    input_dict = {"expected_cardinality": int(42)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A different positive integer
    input_dict = {"expected_cardinality": int(1234)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Minimum integer value within int32 range
    input_dict = {"expected_cardinality": int(np.iinfo(np.int32).min // 10000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Maximum integer value within int32 range
    input_dict = {"expected_cardinality": int(np.iinfo(np.int32).max // 10000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: a smaller positive integer
    input_dict = {"expected_cardinality": int(7)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: a different small positive integer
    input_dict = {"expected_cardinality": int(15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 11: A negative integer
    input_dict = {"expected_cardinality": int(-5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
