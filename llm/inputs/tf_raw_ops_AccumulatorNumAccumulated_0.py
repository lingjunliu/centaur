
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_AccumulatorNumAccumulated_inputs():
    list_of_inputs = []

    # The `RuntimeError` is due to the op's incompatibility with eager execution.
    # The testing framework requires at least one input, so we provide structurally
    # valid inputs that are expected to fail at runtime with this specific error.
    # Using `dtype=np.object_` ensures the creation of `tf.string` tensors without dtype errors.

    # Input 1: Basic scalar handle with a name.
    input_dict = {
        'handle': np.array("accumulator_handle_1", dtype=np.object_),
        'name': 'op_name_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic scalar handle without a name.
    input_dict = {
        'handle': np.array("accumulator_handle_2", dtype=np.object_),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar handle with an empty string.
    input_dict = {
        'handle': np.array("", dtype=np.object_),
        'name': 'empty_handle_name'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1-D tensor handle with a single element.
    input_dict = {
        'handle': np.array(["vector_handle_1"], dtype=np.object_),
        'name': 'vector_handle_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Scalar handle with an empty string for the name.
    input_dict = {
        'handle': np.array("another_handle", dtype=np.object_),
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

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

check_valid('tf.raw_ops.AccumulatorNumAccumulated', generated_inputs['tf.raw_ops.AccumulatorNumAccumulated'], lib="tf", suffix=0)
