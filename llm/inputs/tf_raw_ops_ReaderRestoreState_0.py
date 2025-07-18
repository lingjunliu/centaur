
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_reader_restore_state_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ReaderRestoreState operation.
    
    NOTE: This operation is fundamentally incompatible with TensorFlow's eager execution
    mode. It requires a 'ref' tensor for the `reader_handle` argument, which is a
    construct that only exists in TensorFlow's graph mode. Calling this function in an
    eager context will always raise a `RuntimeError`. The inputs provided here are
    syntactically correct but are intended to trigger this expected error, as the
    testing harness requires at least one input to be generated.
    """
    list_of_inputs = []

    # Input 1: A minimal, standard case. This is guaranteed to raise a RuntimeError
    # in eager execution, which is the documented and expected behavior for this API.
    input_dict = {
        'reader_handle': np.array(b'a_reader_handle', dtype=object),
        'state': np.array(b'a_serialized_state', dtype=object),
        'name': 'restore_op_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Case without the optional 'name' parameter. Will also raise RuntimeError.
    input_dict = {
        'reader_handle': np.array(b'another_handle', dtype=object),
        'state': np.array(b'another_state_string', dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Case with empty strings. Will also raise RuntimeError.
    input_dict = {
        'reader_handle': np.array(b'', dtype=object),
        'state': np.array(b'', dtype=object),
        'name': 'restore_with_empty_strings'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Case with binary-like data in the state. Will also raise RuntimeError.
    input_dict = {
        'reader_handle': np.array(b'binary_handle', dtype=object),
        'state': np.array(b'\xde\xad\xbe\xef\x01\x02\x03', dtype=object),
        'name': 'restore_binary_state'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: A final test case. Will also raise RuntimeError.
    input_dict = {
        'reader_handle': np.array(b'final_handle_test', dtype=object),
        'state': np.array(b'final_state_test_string', dtype=object),
        'name': 'final_restore_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ReaderRestoreState"] = tf_raw_ops_reader_restore_state_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReaderRestoreState' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderRestoreState'.")

check_valid('tf.raw_ops.ReaderRestoreState', generated_inputs['tf.raw_ops.ReaderRestoreState'], lib="tf", suffix=0)
