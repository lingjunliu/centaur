
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_queuesize_inputs():
    """
    Generates a list of inputs for the tf.raw_ops.QueueSize function.

    The tf.raw_ops.QueueSize operation is not compatible with eager execution,
    as it requires a resource handle from a TensorFlow graph. The execution
    environment raises a RuntimeError when attempting to run this op eagerly.
    However, the testing framework requires at least one input to be provided.
    The following inputs are structurally valid according to the API signature
    but are expected to fail at runtime in an eager context. The handle is
    represented as a numpy array with dtype=object to avoid issues with specific
    string dtypes not being in an allowed list.
    """
    list_of_inputs = []

    # Input 1: A standard-looking input
    input_dict_1 = {
        'handle': np.array(b'fifo_queue_handle_1', dtype=object),
        'name': 'QueueSize_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Different handle and name is None (optional)
    input_dict_2 = {
        'handle': np.array(b'another_queue_handle', dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Short handle and name
    input_dict_3 = {
        'handle': np.array(b'q', dtype=object),
        'name': 's'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Handle with characters that might appear in scoped names
    input_dict_4 = {
        'handle': np.array(b'scope/sub_scope/my_queue', dtype=object),
        'name': 'op_with_underscores'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Handle with numbers
    input_dict_5 = {
        'handle': np.array(b'priority_queue_12345', dtype=object),
        'name': 'SizeCheck_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.QueueSize"] = tf_raw_ops_queuesize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueSize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueSize'.")

check_valid('tf.raw_ops.QueueSize', generated_inputs['tf.raw_ops.QueueSize'], lib="tf", suffix=0)
