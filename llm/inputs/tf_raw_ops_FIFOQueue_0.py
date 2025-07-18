
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_raw_ops_fifoqueue_inputs():
    list_of_inputs = []

    # Although tf.raw_ops.FIFOQueue is not compatible with eager execution,
    # the testing framework requires inputs to be generated. The following inputs
    # are syntactically and semantically correct according to the API's signature
    # for a graph-based execution context.

    # Input 1: Basic case with a single float component, default parameters.
    input_dict_1 = {
        'component_types': [tf.float32],
        'shapes': [],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'fifo_queue_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Single integer component with a specified shape.
    input_dict_2 = {
        'component_types': [tf.int32],
        'shapes': [[10]],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'fifo_queue_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Two components with specified shapes.
    input_dict_3 = {
        'component_types': [tf.int64, tf.string],
        'shapes': [[], [5]],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'fifo_queue_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Bounded capacity.
    input_dict_4 = {
        'component_types': [tf.bool],
        'shapes': [[2, 2]],
        'capacity': 100,
        'container': '',
        'shared_name': '',
        'name': 'fifo_queue_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Zero capacity.
    input_dict_5 = {
        'component_types': [tf.double],
        'shapes': [],
        'capacity': 0,
        'container': '',
        'shared_name': '',
        'name': 'fifo_queue_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Using a non-empty container.
    input_dict_6 = {
        'component_types': [tf.uint8],
        'shapes': [[128, 128]],
        'capacity': 50,
        'container': 'my_container_1',
        'shared_name': '',
        'name': 'fifo_queue_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Using a non-empty shared_name.
    input_dict_7 = {
        'component_types': [tf.int16],
        'shapes': [],
        'capacity': 20,
        'container': '',
        'shared_name': 'my_shared_queue_1',
        'name': 'fifo_queue_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Both container and shared_name are specified.
    input_dict_8 = {
        'component_types': [tf.float16],
        'shapes': [[32, 32]],
        'capacity': 10,
        'container': 'my_container_2',
        'shared_name': 'my_shared_queue_2',
        'name': 'fifo_queue_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Complex number component.
    input_dict_9 = {
        'component_types': [tf.complex64],
        'shapes': [[4, 4]],
        'capacity': 5,
        'container': '',
        'shared_name': '',
        'name': 'fifo_queue_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Three components with different shapes.
    input_dict_10 = {
        'component_types': [tf.int32, tf.float32, tf.string],
        'shapes': [[1], [2, 2], []],
        'capacity': 15,
        'container': '',
        'shared_name': '',
        'name': 'fifo_queue_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.FIFOQueue"] = tf_raw_ops_fifoqueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FIFOQueue'.")

check_valid('tf.raw_ops.FIFOQueue', generated_inputs['tf.raw_ops.FIFOQueue'], lib="tf", suffix=0)
