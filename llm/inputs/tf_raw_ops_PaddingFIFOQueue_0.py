
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_padding_fifo_queue_inputs():
    """
    This function generates a list of valid inputs for the tf.raw_ops.PaddingFIFOQueue API.
    The tf.raw_ops.PaddingFIFOQueue op is not supported in eager execution, which is the
    default in modern TensorFlow. These inputs are structurally valid according to the
    API's documentation but will raise a RuntimeError if executed in eager mode,
    as observed from the execution logs.
    """
    list_of_inputs = []

    # Input 1: Basic case with a fixed shape and capacity.
    input_dict_1 = {
        'component_types': [np.int32],
        'shapes': [[2, 3]],
        'capacity': 10,
        'container': '',
        'shared_name': '',
        'name': 'fixed_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple components, one with a variable dimension, unlimited capacity.
    input_dict_2 = {
        'component_types': [np.float32, np.string_],
        'shapes': [[-1], []],
        'capacity': -1,
        'container': 'test_container',
        'shared_name': '',
        'name': 'multi_component_var_shape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Single boolean component, default shape, shared name.
    input_dict_3 = {
        'component_types': [np.bool_],
        'shapes': [],
        'capacity': 1,
        'container': '',
        'shared_name': 'shared_bool_queue',
        'name': 'bool_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: A case with a zero dimension for variable size.
    input_dict_4 = {
        'component_types': [np.uint8],
        'shapes': [[10, 0]],
        'capacity': 100,
        'container': '',
        'shared_name': '',
        'name': 'zero_dim_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    return list_of_inputs

generated_inputs["tf.raw_ops.PaddingFIFOQueue"] = tf_raw_ops_padding_fifo_queue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.PaddingFIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PaddingFIFOQueue'.")

check_valid('tf.raw_ops.PaddingFIFOQueue', generated_inputs['tf.raw_ops.PaddingFIFOQueue'], lib="tf", suffix=0)
