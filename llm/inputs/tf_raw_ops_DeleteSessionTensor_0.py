
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DeleteSessionTensor_inputs():
    # This API is stateful. It is designed to operate on a tensor handle
    # that has been previously created within the same session by an op like
    # `GetSessionHandle`. Executing `DeleteSessionTensor` in a stateless
    # context with a static string handle will always raise a
    # `FailedPreconditionError` because the handle does not reference a live
    # tensor. The inputs provided here are syntactically correct according to
    # the API signature but are expected to produce this specific runtime error
    # under the described test conditions.
    list_of_inputs = []

    # Input 1
    input_dict = {
        'handle': np.array('handle_one'),
        'name': 'delete_op_A'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'handle': np.array('temp_tensor_xyz'),
        'name': 'delete_op_B'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        'handle': np.array('_internal_handle_99'),
        'name': 'delete_op_C'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'handle': np.array('dataflow/state/tensor_handle'),
        'name': 'delete_op_D'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        'handle': np.array('cleanup_target'),
        'name': 'delete_op_E'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DeleteSessionTensor"] = tf_raw_ops_DeleteSessionTensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DeleteSessionTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeleteSessionTensor'.")

check_valid('tf.raw_ops.DeleteSessionTensor', generated_inputs['tf.raw_ops.DeleteSessionTensor'], lib="tf", suffix=0)
