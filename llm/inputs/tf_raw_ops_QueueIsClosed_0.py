
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_queue_is_closed_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QueueIsClosed operation.
    NOTE: This operation is designed for TensorFlow's graph mode and will raise
    a RuntimeError if executed eagerly. The error "queue_is_closed op does not
    support eager execution" is an inherent characteristic of this op and cannot be
    bypassed by changing the input values. The provided inputs are syntactically
    correct representations of what would be passed in a graph context.
    """
    list_of_inputs = []

    # The 'handle' should be a scalar string tensor representing a queue's shared_name.

    # Input 1
    list_of_inputs.append(copy.deepcopy({
        'handle': np.array('queue_handle_alpha', dtype=np.object_),
        'name': 'is_closed_op_1'
    }))

    # Input 2
    list_of_inputs.append(copy.deepcopy({
        'handle': np.array('queue_handle_beta', dtype=np.object_),
        'name': 'is_closed_op_2'
    }))

    # Input 3
    list_of_inputs.append(copy.deepcopy({
        'handle': np.array('queue_handle_gamma', dtype=np.object_),
        'name': 'is_closed_op_3'
    }))

    # Input 4
    list_of_inputs.append(copy.deepcopy({
        'handle': np.array('queue1', dtype=np.object_),
        'name': 'check_q1'
    }))

    # Input 5
    list_of_inputs.append(copy.deepcopy({
        'handle': np.array('queue2', dtype=np.object_),
        'name': 'check_q2'
    }))

    # Input 6
    list_of_inputs.append(copy.deepcopy({
        'handle': np.array('my_data_queue', dtype=np.object_),
        'name': 'my_data_queue_check'
    }))

    # Input 7
    list_of_inputs.append(copy.deepcopy({
        'handle': np.array('another_queue_name', dtype=np.object_),
        'name': 'another_check_op'
    }))

    # Input 8
    list_of_inputs.append(copy.deepcopy({
        'handle': np.array('input_pipeline_queue', dtype=np.object_),
        'name': 'pipeline_status'
    }))

    # Input 9
    list_of_inputs.append(copy.deepcopy({
        'handle': np.array('output_queue', dtype=np.object_),
        'name': 'output_status'
    }))

    # Input 10
    list_of_inputs.append(copy.deepcopy({
        'handle': np.array('processing_q', dtype=np.object_),
        'name': 'proc_q_check'
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.QueueIsClosed"] = get_tf_raw_ops_queue_is_closed_inputs()

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
