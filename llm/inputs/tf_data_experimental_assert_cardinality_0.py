
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_assert_cardinality_inputs():
    list_of_inputs = []

    # Input 1: Positive integer
    expected_cardinality = np.int32(10)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    expected_cardinality = np.int32(0)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Large positive integer
    expected_cardinality = np.int64(100000)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Another positive integer
    expected_cardinality = np.int32(1)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different positive integer
    expected_cardinality = np.int64(42)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small integer
    expected_cardinality = np.int32(2)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Medium integer
    expected_cardinality = np.int64(128)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another larger integer
    expected_cardinality = np.int32(512)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A prime number
    expected_cardinality = np.int64(7)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another composite number
    expected_cardinality = np.int32(15)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Max int32
    expected_cardinality = np.int32(2147483647)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Max int64
    expected_cardinality = np.int64(9223372036854775807)
    input_dict = {"expected_cardinality": expected_cardinality}
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
