
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_ReaderReadUpTo_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ReaderReadUpTo operation.
    NOTE: This operation is designed for TensorFlow's graph mode and will raise
    a RuntimeError in eager execution. The inputs provided are syntactically
    valid according to the API signature, but are expected to fail at runtime
    in an eager context. This is the correct behavior for this specific op.
    """
    list_of_inputs = []

    # Input 1: Basic case
    input_dict_1 = {
        'reader_handle': np.array(b'reader_handle_1', dtype=object),
        'queue_handle': np.array(b'queue_handle_1', dtype=object),
        'num_records': np.array(10, dtype=np.int64),
        'name': 'read_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Read a single record
    input_dict_2 = {
        'reader_handle': np.array(b'reader_handle_2', dtype=object),
        'queue_handle': np.array(b'queue_handle_2', dtype=object),
        'num_records': np.array(1, dtype=np.int64),
        'name': 'read_one'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Read a large number of records
    input_dict_3 = {
        'reader_handle': np.array(b'reader_handle_3', dtype=object),
        'queue_handle': np.array(b'queue_handle_3', dtype=object),
        'num_records': np.array(1000, dtype=np.int64),
        'name': 'read_1000'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: No name provided
    input_dict_4 = {
        'reader_handle': np.array(b'reader_handle_4', dtype=object),
        'queue_handle': np.array(b'queue_handle_4', dtype=object),
        'num_records': np.array(5, dtype=np.int64),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Read zero records
    input_dict_5 = {
        'reader_handle': np.array(b'reader_handle_5', dtype=object),
        'queue_handle': np.array(b'queue_handle_5', dtype=object),
        'num_records': np.array(0, dtype=np.int64),
        'name': 'read_zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Common batch size
    input_dict_6 = {
        'reader_handle': np.array(b'TFRecordReader', dtype=object),
        'queue_handle': np.array(b'FIFOQueue', dtype=object),
        'num_records': np.array(32, dtype=np.int64),
        'name': 'batch_read_32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty handle strings
    input_dict_7 = {
        'reader_handle': np.array(b'', dtype=object),
        'queue_handle': np.array(b'', dtype=object),
        'num_records': np.array(8, dtype=np.int64),
        'name': 'read_with_empty_handles'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Another common batch size
    input_dict_8 = {
        'reader_handle': np.array(b'another_reader', dtype=object),
        'queue_handle': np.array(b'another_queue', dtype=object),
        'num_records': np.array(64, dtype=np.int64),
        'name': 'read_batch_64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Using numbers in handle strings
    input_dict_9 = {
        'reader_handle': np.array(b'reader_999', dtype=object),
        'queue_handle': np.array(b'queue_123', dtype=object),
        'num_records': np.array(25, dtype=np.int64),
        'name': 'read_25'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Max int64 value for num_records
    input_dict_10 = {
        'reader_handle': np.array(b'max_reader', dtype=object),
        'queue_handle': np.array(b'max_queue', dtype=object),
        'num_records': np.array(np.iinfo(np.int64).max, dtype=np.int64),
        'name': 'read_max_records'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
