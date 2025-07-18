
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def tf_queue_fifoqueue_inputs():
    """
    Generates a list of valid inputs for the tf.queue.FIFOQueue function.
    """
    list_of_inputs = []

    # Input 1: Basic case with a single scalar float
    input_dict_1 = {
        'capacity': 10,
        'dtypes': [np.float32],
        'shapes': [[]],
        'names': [b''],
        'shared_name': '',
        'name': 'basic_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple dtypes and shapes
    input_dict_2 = {
        'capacity': 5,
        'dtypes': [np.int32, np.string_],
        'shapes': [[2, 2], []],
        'names': [b'', b''],
        'shared_name': '',
        'name': 'multi_type_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using the 'names' parameter
    input_dict_3 = {
        'capacity': 20,
        'dtypes': [np.float64, np.int64],
        'shapes': [[10], [10]],
        'names': [b'feature', b'label'],
        'shared_name': '',
        'name': 'named_components_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using a 'shared_name'
    input_dict_4 = {
        'capacity': 100,
        'dtypes': [np.bool_],
        'shapes': [[]],
        'names': [b''],
        'shared_name': 'my_shared_queue',
        'name': 'queue_instance_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Minimal capacity
    input_dict_5 = {
        'capacity': 1,
        'dtypes': [np.int8],
        'shapes': [[100, 100]],
        'names': [b''],
        'shared_name': '',
        'name': 'single_capacity_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Higher-dimensional shape (3D tensor)
    input_dict_6 = {
        'capacity': 50,
        'dtypes': [np.float32],
        'shapes': [[2, 3, 4]],
        'names': [b''],
        'shared_name': '',
        'name': '3d_tensor_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Unspecified shapes (shapes=None)
    input_dict_7 = {
        'capacity': 15,
        'dtypes': [np.float32, np.int32],
        'shapes': None,
        'names': [b'data1', b'data2'],
        'shared_name': '',
        'name': 'unspecified_shape_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: All optional parameters provided
    input_dict_8 = {
        'capacity': 8,
        'dtypes': [np.complex64, np.string_],
        'shapes': [[], [5]],
        'names': [b'complex_data', b'metadata'],
        'shared_name': 'my_shared_complex_queue',
        'name': 'full_spec_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Single component, vector shape
    input_dict_9 = {
        'capacity': 32,
        'dtypes': [np.uint8],
        'shapes': [[128]],
        'names': [b''],
        'shared_name': '',
        'name': 'image_row_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Multiple components with mixed shapes and dtypes
    input_dict_10 = {
        'capacity': 25,
        'dtypes': [np.float16, np.int16, np.float32],
        'shapes': [[1], [2, 2], [3, 3, 3]],
        'names': [b'id', b'matrix', b'cube'],
        'shared_name': '',
        'name': 'mixed_shape_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Large capacity queue
    input_dict_11 = {
        'capacity': 10000,
        'dtypes': [np.string_],
        'shapes': [[]],
        'names': [b''],
        'shared_name': '',
        'name': 'large_capacity_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Multiple component queue with different ranks
    input_dict_12 = {
        'capacity': 5,
        'dtypes': [np.int32, np.float32, np.string_, np.bool_],
        'shapes': [[], [2], [5, 5], [1, 2, 3]],
        'names': [b'id', b'embedding', b'image_patch', b'mask'],
        'shared_name': '',
        'name': 'four_component_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))


    return list_of_inputs

generated_inputs["tf.queue.FIFOQueue"] = tf_queue_fifoqueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.FIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.FIFOQueue'.")

check_valid('tf.queue.FIFOQueue', generated_inputs['tf.queue.FIFOQueue'], lib="tf", suffix=0)
