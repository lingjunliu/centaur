
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_queue_close_inputs():
    list_of_inputs = []
    
    # The API tf.raw_ops.QueueClose is fundamentally incompatible with eager 
    # execution. It is a legacy operation from TensorFlow 1 that expects a 
    # reference to a queue handle created within a graph. Calling it in an eager 
    # context will always result in a RuntimeError. The inputs provided here are
    # syntactically valid according to the API's signature but are expected
    # to fail at runtime in an eager environment.

    # Input 1
    input_dict_1 = {
        'handle': np.array("my_test_queue_handle_1", dtype=object),
        'cancel_pending_enqueues': False,
        'name': 'test_close_op_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2
    input_dict_2 = {
        'handle': np.array("my_test_queue_handle_2", dtype=object),
        'cancel_pending_enqueues': True,
        'name': 'test_close_op_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3
    input_dict_3 = {
        'handle': np.array("fifo_queue_resource", dtype=object),
        'cancel_pending_enqueues': False,
        'name': 'close_fifo'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4
    input_dict_4 = {
        'handle': np.array("padding_fifo_queue_resource", dtype=object),
        'cancel_pending_enqueues': True,
        'name': 'close_padding_fifo_cancel'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.QueueClose"] = get_tf_raw_ops_queue_close_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QueueClose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QueueClose'.")

check_valid('tf.raw_ops.QueueClose', generated_inputs['tf.raw_ops.QueueClose'], lib="tf", suffix=0)
