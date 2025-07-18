
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_queue_RandomShuffleQueue_inputs():
    """
    Generates a list of valid inputs for the tf.queue.RandomShuffleQueue function.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single float type and 2D shape
    input_dict_1 = {
        'capacity': 100,
        'min_after_dequeue': 10,
        'dtypes': [np.float32],
        'shapes': [[3, 2]],
        'names': None,
        'seed': 42,
        'shared_name': None,
        'name': 'basic_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple dtypes and fully specified shapes
    input_dict_2 = {
        'capacity': 200,
        'min_after_dequeue': 50,
        'dtypes': [np.int32, np.string_],
        'shapes': [[4], []],
        'names': None,
        'seed': 123,
        'shared_name': 'shared_queue_2',
        'name': 'multi_type_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Unknown shapes for multiple dtypes
    input_dict_3 = {
        'capacity': 50,
        'min_after_dequeue': 0,
        'dtypes': [np.bool_, np.complex64],
        'shapes': [None, None],
        'names': None,
        'seed': None,
        'shared_name': None,
        'name': 'unknown_shape_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Partially unknown shapes
    input_dict_4 = {
        'capacity': 150,
        'min_after_dequeue': 75,
        'dtypes': [np.float64, np.int8],
        'shapes': [[None, 5], [2]],
        'names': None,
        'seed': 2024,
        'shared_name': None,
        'name': 'partial_shape_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Minimal configuration with scalar shape
    input_dict_5 = {
        'capacity': 10,
        'min_after_dequeue': 1,
        'dtypes': [np.int8],
        'shapes': [[]],
        'names': None,
        'seed': None,
        'shared_name': None,
        'name': 'minimal_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Large capacity and mixed shapes
    input_dict_6 = {
        'capacity': 10000,
        'min_after_dequeue': 9000,
        'dtypes': [np.uint16, np.float16],
        'shapes': [[], [10, 10]],
        'names': None,
        'seed': -1,
        'shared_name': 'large_shared_queue',
        'name': 'large_cap_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: min_after_dequeue is zero
    input_dict_7 = {
        'capacity': 30,
        'min_after_dequeue': 0,
        'dtypes': [np.int64],
        'shapes': [[1]],
        'names': None,
        'seed': 777,
        'shared_name': None,
        'name': 'zero_min_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: High-dimensional tensor shape
    input_dict_8 = {
        'capacity': 500,
        'min_after_dequeue': 100,
        'dtypes': [np.float32],
        'shapes': [[2, 3, 4, 5, 6]],
        'names': None,
        'seed': 2023,
        'shared_name': None,
        'name': 'high_dim_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: All optional arguments provided
    input_dict_9 = {
        'capacity': 128,
        'min_after_dequeue': 64,
        'dtypes': [np.string_],
        'shapes': [[4, 4]],
        'names': None,
        'seed': 98765,
        'shared_name': 'fully_specified_queue',
        'name': 'all_args_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Negative seed value and another data type
    input_dict_10 = {
        'capacity': 256,
        'min_after_dequeue': 128,
        'dtypes': [np.int64],
        'shapes': [[8, 8, 8]],
        'names': None,
        'seed': -10,
        'shared_name': None,
        'name': 'neg_seed_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.queue.RandomShuffleQueue"] = tf_queue_RandomShuffleQueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.RandomShuffleQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.RandomShuffleQueue'.")

check_valid('tf.queue.RandomShuffleQueue', generated_inputs['tf.queue.RandomShuffleQueue'], lib="tf", suffix=0)
