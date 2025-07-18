
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_queue_paddingfifoqueue_inputs():
    """
    Generates a list of valid inputs for the tf.queue.PaddingFIFOQueue constructor.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single component and a fixed shape.
    input_dict_1 = {
        'capacity': 10,
        'dtypes': [np.int32],
        'shapes': [[2, 2]],
        'names': None,
        'shared_name': 'q1',
        'name': 'queue_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Two components with different dtypes and fixed shapes.
    input_dict_2 = {
        'capacity': 50,
        'dtypes': [np.float32, np.int64],
        'shapes': [[3], []],
        'names': None,
        'shared_name': 'q2',
        'name': 'float_int_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Single component with one dynamic dimension.
    input_dict_3 = {
        'capacity': 20,
        'dtypes': [np.float64],
        'shapes': [[None, 5]],
        'names': None,
        'shared_name': 'q3',
        'name': 'queue_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Two components, one fixed shape, one with a dynamic dimension.
    input_dict_4 = {
        'capacity': 100,
        'dtypes': [np.int64, np.float32],
        'shapes': [[10], [None, 3, 4]],
        'names': None,
        'shared_name': 'q4',
        'name': 'queue_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Multiple components with mixed dtypes and various dynamic shapes.
    input_dict_5 = {
        'capacity': 30,
        'dtypes': [np.int32, np.bool_],
        'shapes': [[None], [None, None]],
        'names': None,
        'shared_name': 'q5',
        'name': 'mixed_dynamic_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Another two-component example with empty shared_name.
    input_dict_6 = {
        'capacity': 5,
        'dtypes': [np.float32, np.int16],
        'shapes': [[None, 2], [4]],
        'names': None,
        'shared_name': '',
        'name': 'queue_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Using a specific shared_name and a custom name.
    input_dict_7 = {
        'capacity': 1000,
        'dtypes': [np.float32],
        'shapes': [[None]],
        'names': None,
        'shared_name': 'my_shared_queue',
        'name': 'custom_queue_name'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Various integer types.
    input_dict_8 = {
        'capacity': 25,
        'dtypes': [np.int8, np.uint8],
        'shapes': [[1, None], [2, None]],
        'names': None,
        'shared_name': 'image_queue',
        'name': 'image_padding_fifo_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: High-rank tensors with dynamic dimensions.
    input_dict_9 = {
        'capacity': 15,
        'dtypes': [np.float64, np.int64],
        'shapes': [[None, 2, 3, 4], [5, None, 6]],
        'names': None,
        'shared_name': 'q9',
        'name': 'queue_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Queue for complex numbers.
    input_dict_10 = {
        'capacity': 12,
        'dtypes': [np.complex64, np.complex128],
        'shapes': [[None, 64], [None, 128]],
        'names': None,
        'shared_name': 'fft_queue',
        'name': 'queue_11'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.queue.PaddingFIFOQueue"] = tf_queue_paddingfifoqueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.PaddingFIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.PaddingFIFOQueue'.")

check_valid('tf.queue.PaddingFIFOQueue', generated_inputs['tf.queue.PaddingFIFOQueue'], lib="tf", suffix=0)
