
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_assert_cardinality_inputs():
    list_of_inputs = []

    # Input 1: Positive integer
    expected_cardinality = np.int64(10)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    expected_cardinality = np.int32(0)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Large positive integer
    expected_cardinality = np.int64(2**31 - 1)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive integer
    expected_cardinality = np.int32(1)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Another positive integer
    expected_cardinality = np.int64(1000)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.int8
    expected_cardinality = np.int8(50)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: np.int16
    expected_cardinality = np.int16(500)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: np.uint8
    expected_cardinality = np.uint8(200)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: np.uint16
    expected_cardinality = np.uint16(5000)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Another positive integer with different value.
    expected_cardinality = np.int64(12345)
    input_dict = {"expected_cardinality": expected_cardinality}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.assert_cardinality', generated_inputs['tf.data.experimental.assert_cardinality'], lib="tf", suffix=0)
