
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_assert_cardinality_inputs():
    list_of_inputs = []

    # Input 1: Positive integer
    expected_cardinality = np.int64(10)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    expected_cardinality = np.int64(0)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A small positive integer
    expected_cardinality = np.int64(20)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive integer
    expected_cardinality = np.int64(1)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Another positive integer
    expected_cardinality = np.int64(100)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A different positive number
    expected_cardinality = np.int64(50)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: A medium number
    expected_cardinality = np.int64(500)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: A different small number
    expected_cardinality = np.int64(2)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: A positive integer
    expected_cardinality = np.int64(55)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another integer.
    expected_cardinality = np.int64(123)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Larger int
    expected_cardinality = np.int64(10000)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Odd number
    expected_cardinality = np.int64(99)
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
