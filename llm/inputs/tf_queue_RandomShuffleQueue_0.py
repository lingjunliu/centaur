
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

    # Input 1: Basic case
    input_dict_1 = {
        'capacity': 10,
        'min_after_dequeue': 5,
        'dtypes': [np.float32],
        'shapes': [[2, 2]],
        'names': ['float_tensor'],
        'seed': 42,
        'shared_name': 'q1',
        'name': 'basic_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Integer data type and a vector shape.
    input_dict_2 = {
        'capacity': 20,
        'min_after_dequeue': 10,
        'dtypes': [np.int32],
        'shapes': [[3]],
        'names': ['int_vector'],
        'seed': 123,
        'shared_name': 'q2',
        'name': 'int_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: String data type and scalar shape.
    input_dict_3 = {
        'capacity': 50,
        'min_after_dequeue': 0,
        'dtypes': [np.string_],
        'shapes': [[]],
        'names': ['string_scalar'],
        'seed': 0,
        'shared_name': 'q3',
        'name': 'string_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Boolean data type, high capacity.
    input_dict_4 = {
        'capacity': 1000,
        'min_after_dequeue': 500,
        'dtypes': [np.bool_],
        'shapes': [[10, 10]],
        'names': ['bool_matrix'],
        'seed': 1,
        'shared_name': 'q4',
        'name': 'bool_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Complex data type.
    input_dict_5 = {
        'capacity': 15,
        'min_after_dequeue': 7,
        'dtypes': [np.complex64],
        'shapes': [[4]],
        'names': ['complex_tensor'],
        'seed': 99,
        'shared_name': 'q5',
        'name': 'complex_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Edge case: min_after_dequeue is capacity - 1
    input_dict_6 = {
        'capacity': 30,
        'min_after_dequeue': 29,
        'dtypes': [np.int16],
        'shapes': [[5, 5, 5]],
        'names': ['int16_cube'],
        'seed': 42,
        'shared_name': 'q6',
        'name': 'high_water_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: uint8, empty shared_name.
    input_dict_7 = {
        'capacity': 5,
        'min_after_dequeue': 2,
        'dtypes': [np.uint8],
        'shapes': [[100]],
        'names': ['uint8_vec'],
        'seed': 0,
        'shared_name': '',
        'name': 'no_shared_name_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: double (float64), large capacity.
    input_dict_8 = {
        'capacity': 10000,
        'min_after_dequeue': 5000,
        'dtypes': [np.float64],
        'shapes': [[128, 128]],
        'names': ['large_image'],
        'seed': 2023,
        'shared_name': 'q8',
        'name': 'large_cap_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Minimal capacity.
    input_dict_9 = {
        'capacity': 1,
        'min_after_dequeue': 0,
        'dtypes': [np.int64],
        'shapes': [[1]],
        'names': ['minimal_queue_tensor'],
        'seed': 7,
        'shared_name': 'q9',
        'name': 'minimal_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: High dimensional shape.
    input_dict_10 = {
        'capacity': 40,
        'min_after_dequeue': 20,
        'dtypes': [np.float16],
        'shapes': [[2, 3, 4, 5]],
        'names': ['high_dim_tensor'],
        'seed': 88,
        'shared_name': 'q10',
        'name': 'high_dim_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Multiple dtypes and names.
    input_dict_11 = {
        'capacity': 25,
        'min_after_dequeue': 12,
        'dtypes': [np.int32, np.float32, np.string_],
        'shapes': [[2], [2], []],
        'names': ['ints', 'floats', 'strings'],
        'seed': 101,
        'shared_name': 'q11',
        'name': 'multi_type_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))


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
