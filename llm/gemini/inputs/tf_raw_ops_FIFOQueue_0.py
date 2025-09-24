
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fifoqueue_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.FIFOQueue function.
    NOTE: This operation is designed for TensorFlow's graph mode and will raise a
    RuntimeError if executed eagerly, as it returns a reference handle. The generated
    inputs are valid for graph construction.
    """
    list_of_inputs = []

    # Input 1: Simplest case - float, no shape constraint, infinite capacity.
    input_dict_1 = {
        'component_types': [tf.float32],
        'shapes': [],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'simple_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Integer queue with a fixed capacity and a defined 1D shape.
    input_dict_2 = {
        'component_types': [tf.int32],
        'shapes': [[10]],
        'capacity': 100,
        'container': '',
        'shared_name': '',
        'name': 'int_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Queue for two components (e.g., data and label).
    input_dict_3 = {
        'component_types': [tf.float64, tf.int64],
        'shapes': [[224, 224], []],
        'capacity': 50,
        'container': '',
        'shared_name': '',
        'name': 'data_label_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: String queue with resource management (container and shared_name).
    input_dict_4 = {
        'component_types': [tf.string],
        'shapes': [[]],
        'capacity': 200,
        'container': 'my_app_container',
        'shared_name': 'shared_string_queue',
        'name': 'resource_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Zero-capacity queue (acts as a rendezvous/synchronization point).
    input_dict_5 = {
        'component_types': [tf.bool],
        'shapes': [[1]],
        'capacity': 0,
        'container': '',
        'shared_name': '',
        'name': 'sync_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Complex number queue.
    input_dict_6 = {
        'component_types': [tf.complex128],
        'shapes': [[64, 64]],
        'capacity': 10,
        'container': '',
        'shared_name': '',
        'name': 'complex_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: High-dimensional tensor queue (e.g., for video frames).
    input_dict_7 = {
        'component_types': [tf.uint8],
        'shapes': [[32, 240, 320, 3]],
        'capacity': 5,
        'container': '',
        'shared_name': '',
        'name': 'video_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: bfloat16 queue.
    input_dict_8 = {
        'component_types': [tf.bfloat16],
        'shapes': [[1024]],
        'capacity': 128,
        'container': '',
        'shared_name': '',
        'name': 'bfloat_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Multiple components with no shape constraints.
    input_dict_9 = {
        'component_types': [tf.int8, tf.float16, tf.string],
        'shapes': [],
        'capacity': 64,
        'container': '',
        'shared_name': '',
        'name': 'multi_unshaped_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Using tf.TensorShape objects for shapes.
    input_dict_10 = {
        'component_types': [tf.float32, tf.int32],
        'shapes': [tf.TensorShape([None, 10]), tf.TensorShape([None])],
        'capacity': 32,
        'container': '',
        'shared_name': '',
        'name': 'partial_shape_queue'
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
