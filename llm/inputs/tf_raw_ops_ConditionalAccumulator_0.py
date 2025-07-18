
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_conditionalaccumulator_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ConditionalAccumulator function.
    NOTE: This op is not compatible with eager execution and is designed for TensorFlow's
    graph mode. Calling it in an eager context will raise a RuntimeError. The inputs
    provided are valid with respect to the function's signature for a graph context.
    """
    list_of_inputs = []

    # Input 1: Basic float32, MEAN reduction
    input_dict_1 = {
        'dtype': tf.float32,
        'shape': [10],
        'container': '',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'acc_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: int32, SUM reduction, 2D shape
    input_dict_2 = {
        'dtype': tf.int32,
        'shape': [5, 5],
        'container': '',
        'shared_name': '',
        'reduction_type': 'SUM',
        'name': 'acc_int32_sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: float64, MEAN reduction, scalar shape
    input_dict_3 = {
        'dtype': tf.float64,
        'shape': [],
        'container': '',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'acc_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: int64, SUM, with container and shared_name
    input_dict_4 = {
        'dtype': tf.int64,
        'shape': [100],
        'container': 'my_container',
        'shared_name': 'my_shared_name',
        'reduction_type': 'SUM',
        'name': 'acc_shared'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: uint8, MEAN, 3D shape
    input_dict_5 = {
        'dtype': tf.uint8,
        'shape': [2, 3, 4],
        'container': '',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'acc_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: half dtype
    input_dict_6 = {
        'dtype': tf.half,
        'shape': [16],
        'container': '',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'acc_half'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: bfloat16 dtype
    input_dict_7 = {
        'dtype': tf.bfloat16,
        'shape': [32, 32],
        'container': '',
        'shared_name': 'bfloat_shared',
        'reduction_type': 'SUM',
        'name': 'acc_bfloat16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: int8 dtype
    input_dict_8 = {
        'dtype': tf.int8,
        'shape': [256],
        'container': 'int8_container',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'acc_int8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: uint16 dtype
    input_dict_9 = {
        'dtype': tf.uint16,
        'shape': [4, 8, 16],
        'container': '',
        'shared_name': 'uint16_shared',
        'reduction_type': 'SUM',
        'name': 'acc_uint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: complex64 dtype
    input_dict_10 = {
        'dtype': tf.complex64,
        'shape': [4, 4],
        'container': 'complex_container',
        'shared_name': 'complex_shared',
        'reduction_type': 'MEAN',
        'name': 'acc_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ConditionalAccumulator"] = get_tf_raw_ops_conditionalaccumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ConditionalAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ConditionalAccumulator'.")

check_valid('tf.raw_ops.ConditionalAccumulator', generated_inputs['tf.raw_ops.ConditionalAccumulator'], lib="tf", suffix=0)
