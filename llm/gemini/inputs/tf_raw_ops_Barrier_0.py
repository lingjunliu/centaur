
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.Barrier.
    Note: This operation is designed for TensorFlow's graph execution mode and
    is expected to raise a RuntimeError when executed eagerly, which is the
    default in TensorFlow 2.x. The inputs provided are syntactically correct
    according to the API's signature.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single float component, default everything else.
    input_dict_1 = {
        'component_types': [np.float32],
        'shapes': [],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'barrier_np_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple component types.
    input_dict_2 = {
        'component_types': [np.int64, np.string_, np.bool_],
        'shapes': [],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'barrier_np_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With specified shapes (as list of lists of ints).
    input_dict_3 = {
        'component_types': [np.int32, np.float32],
        'shapes': [[1, 8], [1, 4, 4]],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'barrier_np_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With a specific positive capacity.
    input_dict_4 = {
        'component_types': [np.complex64],
        'shapes': [[1, 10]],
        'capacity': 256,
        'container': '',
        'shared_name': '',
        'name': 'barrier_np_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: With a non-empty container.
    input_dict_5 = {
        'component_types': [np.uint8],
        'shapes': [],
        'capacity': -1,
        'container': 'my_container_np',
        'shared_name': '',
        'name': 'barrier_np_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: With a non-empty shared_name.
    input_dict_6 = {
        'component_types': [np.uint16, np.int16],
        'shapes': [[1, 5], [1, 5]],
        'capacity': 128,
        'container': '',
        'shared_name': 'shared_barrier_np',
        'name': 'barrier_np_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: All optional arguments specified.
    input_dict_7 = {
        'component_types': [np.float64, np.int32],
        'shapes': [[1, 2, 3], [1, 6]],
        'capacity': 20,
        'container': 'another_container_np',
        'shared_name': 'another_shared_barrier_np',
        'name': 'barrier_np_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Mix of various types and shapes.
    input_dict_8 = {
        'component_types': [np.float16, np.complex128, np.int8],
        'shapes': [[1, 50], [1, 3, 3], [1]],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'barrier_np_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: High-dimensional shape and a small capacity.
    input_dict_9 = {
        'component_types': [np.int32],
        'shapes': [[1, 2, 2, 2, 2, 2]],
        'capacity': 2,
        'container': '',
        'shared_name': 'high_dim_barrier_np',
        'name': 'barrier_np_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Longer list of components.
    input_dict_10 = {
        'component_types': [np.float32, np.int32, np.string_, np.bool_, np.complex64],
        'shapes': [[1, 1], [1, 1], [1], [1, 1], [1, 1]],
        'capacity': 50,
        'container': 'long_list_container_np',
        'shared_name': 'long_list_shared_np',
        'name': 'barrier_np_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Barrier"] = tf_raw_ops_barrier_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Barrier' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Barrier'.")

check_valid('tf.raw_ops.Barrier', generated_inputs['tf.raw_ops.Barrier'], lib="tf", suffix=0)
