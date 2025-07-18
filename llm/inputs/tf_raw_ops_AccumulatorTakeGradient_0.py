
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_accumulator_take_gradient_inputs():
    """
    This function generates a list of valid inputs for the
    tf.raw_ops.AccumulatorTakeGradient operation.
    NOTE: This operation is not supported in Eager execution mode and requires
    a valid resource handle created in a graph context. The provided inputs
    are syntactically correct (handle is a scalar object tensor representing a string)
    but are expected to fail if run in a standard eager context, as the op
    explicitly checks for and disallows eager execution.
    """
    list_of_inputs = []

    # The handle is a scalar tensor of type string. We use a scalar numpy array
    # with dtype=object to represent this, as it correctly translates to a
    # tf.string tensor and avoids dtype issues with the testing framework.
    
    # Case 1: Basic float32
    input_dict_1 = {
        'handle': np.array('handle1', dtype=object),
        'num_required': np.array(1, dtype=np.int32),
        'dtype': tf.float32,
        'name': 'take_grad_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: float64 with a larger num_required
    input_dict_2 = {
        'handle': np.array('handle2', dtype=object),
        'num_required': np.array(100, dtype=np.int32),
        'dtype': tf.float64,
        'name': 'take_grad_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: int32
    input_dict_3 = {
        'handle': np.array('handle3', dtype=object),
        'num_required': np.array(5, dtype=np.int32),
        'dtype': tf.int32,
        'name': 'take_grad_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: uint8
    input_dict_4 = {
        'handle': np.array('handle4', dtype=object),
        'num_required': np.array(10, dtype=np.int32),
        'dtype': tf.uint8,
        'name': 'take_grad_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: complex64
    input_dict_5 = {
        'handle': np.array('handle5', dtype=object),
        'num_required': np.array(2, dtype=np.int32),
        'dtype': tf.complex64,
        'name': 'take_grad_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: int64
    input_dict_6 = {
        'handle': np.array('handle6', dtype=object),
        'num_required': np.array(50, dtype=np.int32),
        'dtype': tf.int64,
        'name': 'take_grad_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: half (float16)
    input_dict_7 = {
        'handle': np.array('handle7', dtype=object),
        'num_required': np.array(1, dtype=np.int32),
        'dtype': tf.half,
        'name': 'take_grad_half'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: complex128
    input_dict_8 = {
        'handle': np.array('handle8', dtype=object),
        'num_required': np.array(3, dtype=np.int32),
        'dtype': tf.complex128,
        'name': 'take_grad_complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: No name provided (optional)
    input_dict_9 = {
        'handle': np.array('handle9', dtype=object),
        'num_required': np.array(1000, dtype=np.int32),
        'dtype': tf.uint32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: int16
    input_dict_10 = {
        'handle': np.array('handle10', dtype=object),
        'num_required': np.array(256, dtype=np.int32),
        'dtype': tf.int16,
        'name': 'take_grad_int16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.AccumulatorTakeGradient"] = tf_raw_ops_accumulator_take_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulatorTakeGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulatorTakeGradient'.")

check_valid('tf.raw_ops.AccumulatorTakeGradient', generated_inputs['tf.raw_ops.AccumulatorTakeGradient'], lib="tf", suffix=0)
