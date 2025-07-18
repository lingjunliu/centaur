
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_inputs_for_tf_raw_ops_QueueIsClosed():
    list_of_inputs = []

    # This raw op is a legacy TF1 operation not compatible with eager execution.
    # It expects a 'ref' handle, which cannot be created from numpy in eager mode.
    # The following inputs are syntactically correct but will cause the
    # documented runtime error in an eager context.

    # Input 1: A scalar numpy array for the handle.
    input_dict = {
        'handle': np.array("queue_handle_1", dtype=np.object_),
        'name': "QueueIsClosed_Test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: A 1-element, 1-D numpy array for the handle.
    input_dict = {
        'handle': np.array(["queue_handle_2"], dtype=np.object_),
        'name': "QueueIsClosed_Test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using a different name for the handle and operation.
    input_dict = {
        'handle': np.array("my_fifo_queue", dtype=np.object_),
        'name': "check_fifo_queue_closed"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Using a byte string for the handle content, wrapped in a scalar array.
    input_dict = {
        'handle': np.array(b"byte_string_handle", dtype=np.object_),
        'name': "QueueIsClosed_Bytes_Test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Optional name is None
    input_dict = {
        'handle': np.array("queue_handle_5", dtype=np.object_),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QueueIsClosed"] = generate_inputs_for_tf_raw_ops_QueueIsClosed()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueIsClosed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueIsClosed'.")

check_valid('tf.raw_ops.QueueIsClosed', generated_inputs['tf.raw_ops.QueueIsClosed'], lib="tf", suffix=0)
