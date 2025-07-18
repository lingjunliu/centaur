
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_ReaderReadUpTo_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ReaderReadUpTo function.
    NOTE: This operation is from TensorFlow V1 and is not compatible with eager
    execution, which is the default in modern TensorFlow. It is expected to
    raise a RuntimeError when called in an eager context. The generated inputs
    are syntactically correct for the API's signature.
    """
    list_of_inputs = []

    # Resource handles are represented as scalar numpy arrays with dtype=object
    # to hold a placeholder string. This format is a valid representation
    # for a scalar string tensor.
    handle_placeholder = np.array("placeholder_handle", dtype=object)

    base_input = {
        'reader_handle': handle_placeholder,
        'queue_handle': handle_placeholder,
    }

    # Case 1: A typical batch size like 16
    input_dict = copy.deepcopy(base_input)
    input_dict['name'] = 'read_batch_of_16'
    input_dict['num_records'] = np.array(16, dtype=np.int64)
    list_of_inputs.append(input_dict)

    # Case 2: Read a single record
    input_dict = copy.deepcopy(base_input)
    input_dict['name'] = 'read_one_item'
    input_dict['num_records'] = np.array(1, dtype=np.int64)
    list_of_inputs.append(input_dict)

    # Case 3: Read zero records, which is a valid edge case
    input_dict = copy.deepcopy(base_input)
    input_dict['name'] = 'read_zero_items'
    input_dict['num_records'] = np.array(0, dtype=np.int64)
    list_of_inputs.append(input_dict)

    # Case 4: No optional name provided (name=None)
    input_dict = copy.deepcopy(base_input)
    input_dict['name'] = None
    input_dict['num_records'] = np.array(64, dtype=np.int64)
    list_of_inputs.append(input_dict)

    # Case 5: A non-power-of-two number of records
    input_dict = copy.deepcopy(base_input)
    input_dict['name'] = 'read_99_items'
    input_dict['num_records'] = np.array(99, dtype=np.int64)
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.ReaderReadUpTo"] = tf_raw_ops_ReaderReadUpTo_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReaderReadUpTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderReadUpTo'.")

check_valid('tf.raw_ops.ReaderReadUpTo', generated_inputs['tf.raw_ops.ReaderReadUpTo'], lib="tf", suffix=0)
