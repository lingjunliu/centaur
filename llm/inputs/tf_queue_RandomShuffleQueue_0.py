
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

    # Input 1: Basic case with a single float tensor type.
    input_dict_1 = {
        'capacity': 100,
        'min_after_dequeue': 10,
        'dtypes': [np.float32],
        'shapes': [[10, 2]],
        'names': [1],
        'seed': 1,
        'shared_name': 'q1',
        'name': 'basic_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple data types. names is a list of integers.
    input_dict_2 = {
        'capacity': 50,
        'min_after_dequeue': 20,
        'dtypes': [np.int64, np.string_],
        'shapes': [[5], []],
        'names': [10, 20],
        'seed': 2,
        'shared_name': 'q2',
        'name': 'multi_dtype_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar string.
    input_dict_3 = {
        'capacity': 25,
        'min_after_dequeue': 5,
        'dtypes': [np.string_],
        'shapes': [[]],
        'names': [3],
        'seed': 3,
        'shared_name': 'q3',
        'name': 'string_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Double-precision float.
    input_dict_4 = {
        'capacity': 20,
        'min_after_dequeue': 5,
        'dtypes': [np.float64],
        'shapes': [[1, 3, 4]],
        'names': [4],
        'seed': 4,
        'shared_name': 'q4',
        'name': 'double_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Scalar int32.
    input_dict_5 = {
        'capacity': 30,
        'min_after_dequeue': 15,
        'dtypes': [np.int32],
        'shapes': [[]],
        'names': [5],
        'seed': 5,
        'shared_name': 'q5',
        'name': 'scalar_int_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Image-like uint8 tensor. min_after_dequeue is 0.
    input_dict_6 = {
        'capacity': 100,
        'min_after_dequeue': 0,
        'dtypes': [np.uint8],
        'shapes': [[28, 28, 3]],
        'names': [6],
        'seed': 42,
        'shared_name': 'q6',
        'name': 'seeded_image_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Large capacity int16 vector.
    input_dict_7 = {
        'capacity': 2000,
        'min_after_dequeue': 1000,
        'dtypes': [np.int16],
        'shapes': [[128]],
        'names': [7],
        'seed': 7,
        'shared_name': 'global_feature_queue',
        'name': 'large_cap_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Multiple components with bool and complex types.
    input_dict_8 = {
        'capacity': 500,
        'min_after_dequeue': 100,
        'dtypes': [np.bool_, np.complex64],
        'shapes': [[], [2, 2]],
        'names': [81, 82],
        'seed': -99,
        'shared_name': 'multi_comp_shared_queue',
        'name': 'boolean_complex_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Complex number matrix.
    input_dict_9 = {
        'capacity': 2,
        'min_after_dequeue': 1,
        'dtypes': [np.complex128],
        'shapes': [[2, 2]],
        'names': [9],
        'seed': 9,
        'shared_name': 'q9',
        'name': 'minimal_complex_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: High-rank float16 tensor.
    input_dict_10 = {
        'capacity': 10000,
        'min_after_dequeue': 5000,
        'dtypes': [np.float16],
        'shapes': [[2, 3, 4, 5, 6]],
        'names': [10],
        'seed': 2023,
        'shared_name': 'q10',
        'name': 'high_rank_queue'
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
