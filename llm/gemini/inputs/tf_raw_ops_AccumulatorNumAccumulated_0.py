
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AccumulatorNumAccumulated_inputs():
    list_of_inputs = []

    # Input 1
    handle = np.array("accumulator_1", dtype=np.string_)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = np.array("accumulator_2", dtype=np.string_)
    input_dict = {"handle": handle, "name": "my_accumulator"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = np.array("another_accumulator", dtype=np.string_)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = np.array("test_accumulator", dtype=np.string_)
    input_dict = {"handle": handle, "name": "test_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = np.array("acc5", dtype=np.string_)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = np.array("very_long_accumulator_name_6", dtype=np.string_)
    input_dict = {"handle": handle, "name": "long_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = np.array("acc7", dtype=np.string_)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = np.array("acc8", dtype=np.string_)
    input_dict = {"handle": handle, "name": "acc8_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    handle = np.array("a9", dtype=np.string_)
    input_dict = {"handle": handle, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = np.array("a10", dtype=np.string_)
    input_dict = {"handle": handle, "name": "a10_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulatorNumAccumulated"] = tf_raw_ops_AccumulatorNumAccumulated_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulatorNumAccumulated' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorNumAccumulated'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.AccumulatorNumAccumulated', generated_inputs['tf.raw_ops.AccumulatorNumAccumulated'], lib="tf", suffix=0)
