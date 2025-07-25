
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_randomshufflequeue_inputs():
    """
    This function generates a list of valid inputs for the tf.raw_ops.RandomShuffleQueue API.
    Note: This raw op is designed for TensorFlow's graph mode and will raise a RuntimeError
    if called directly under eager execution, as it creates a stateful resource handle.
    The provided inputs are valid for the API's signature in a graph-based context.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single float component.
    input_dict_1 = {
        'component_types': [tf.float32],
        'shapes': [[10]],
        'capacity': 100,
        'min_after_dequeue': 10,
        'seed': 0,
        'seed2': 0,
        'container': '',
        'shared_name': '',
        'name': 'queue_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple components with different types and shapes.
    input_dict_2 = {
        'component_types': [tf.int64, tf.string],
        'shapes': [[5], [1]],
        'capacity': 50,
        'min_after_dequeue': 5,
        'seed': 1,
        'seed2': 2,
        'container': '',
        'shared_name': '',
        'name': 'queue_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Boolean type, no limit capacity.
    input_dict_3 = {
        'component_types': [tf.bool],
        'shapes': [[100, 100]],
        'capacity': -1,
        'min_after_dequeue': 100,
        'seed': 42,
        'seed2': 42,
        'container': '',
        'shared_name': '',
        'name': 'queue_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Shared queue with container and shared_name.
    input_dict_4 = {
        'component_types': [tf.float16],
        'shapes': [[32, 32, 3]],
        'capacity': 1000,
        'min_after_dequeue': 200,
        'seed': 0,
        'seed2': 0,
        'container': 'my_app_container',
        'shared_name': 'shared_image_queue',
        'name': 'queue_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Complex number type.
    input_dict_5 = {
        'component_types': [tf.complex64],
        'shapes': [[4, 8]],
        'capacity': 20,
        'min_after_dequeue': 0,
        'seed': 1234,
        'seed2': 0,
        'container': '',
        'shared_name': '',
        'name': 'queue_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Unconstrained shapes for queue elements.
    input_dict_6 = {
        'component_types': [tf.int32],
        'shapes': [],
        'capacity': 64,
        'min_after_dequeue': 16,
        'seed': 0,
        'seed2': 0,
        'container': '',
        'shared_name': '',
        'name': 'queue_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Minimal valid capacity.
    input_dict_7 = {
        'component_types': [tf.uint8],
        'shapes': [[128]],
        'capacity': 2,
        'min_after_dequeue': 1,
        'seed': 7,
        'seed2': 8,
        'container': '',
        'shared_name': '',
        'name': 'queue_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Different integer types.
    input_dict_8 = {
        'component_types': [tf.int8, tf.int16, tf.uint16],
        'shapes': [[1], [2], [3]],
        'capacity': 30,
        'min_after_dequeue': 3,
        'seed': 99,
        'seed2': 99,
        'container': '',
        'shared_name': '',
        'name': 'queue_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large capacity and min_after_dequeue.
    input_dict_9 = {
        'component_types': [tf.float32],
        'shapes': [[256, 256]],
        'capacity': 5000,
        'min_after_dequeue': 1000,
        'seed': 0,
        'seed2': 0,
        'container': '',
        'shared_name': '',
        'name': 'queue_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: bfloat16 type.
    input_dict_10 = {
        'component_types': [tf.bfloat16],
        'shapes': [[16, 16]],
        'capacity': 128,
        'min_after_dequeue': 64,
        'seed': 2024,
        'seed2': 2025,
        'container': '',
        'shared_name': '',
        'name': 'queue_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomShuffleQueue"] = tf_raw_ops_randomshufflequeue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RandomShuffleQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomShuffleQueue'.")

check_valid('tf.raw_ops.RandomShuffleQueue', generated_inputs['tf.raw_ops.RandomShuffleQueue'], lib="tf", suffix=0)
