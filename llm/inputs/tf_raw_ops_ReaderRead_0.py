
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_reader_read_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ReaderRead function.
    """
    list_of_inputs = []

    # The op this function calls is not compatible with eager execution.
    # The generated inputs are valid for a graph context but will fail
    # when run eagerly, as is done in the testing environment.
    # This is an unavoidable consequence of the API's design.

    for i in range(10):
        reader_handle_val = f"reader_handle_v{i}"
        queue_handle_val = f"queue_handle_v{i}"
        op_name = f"ReaderRead_{i}" if i % 3 != 0 else None
        
        # Use 'object' dtype for string tensors to avoid numpy deprecation issues
        # and ensure they are treated as scalar string tensors by TensorFlow.
        dtype = object

        input_dict = {
            'reader_handle': np.array(reader_handle_val, dtype=dtype),
            'queue_handle': np.array(queue_handle_val, dtype=dtype),
            'name': op_name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ReaderRead"] = tf_raw_ops_reader_read_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReaderRead' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReaderRead'.")

check_valid('tf.raw_ops.ReaderRead', generated_inputs['tf.raw_ops.ReaderRead'], lib="tf", suffix=0)
