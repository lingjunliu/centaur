
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def tf_queue_FIFOQueue_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid case with a single component and a 1-element shape.
    input_dict_1 = {
        'capacity': 10,
        'dtypes': [np.float32],
        'shapes': [(1,)],
        'names': ['value'],
        'shared_name': '',
        'name': 'queue_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple components with different dtypes and non-scalar shapes.
    input_dict_2 = {
        'capacity': 100,
        'dtypes': [np.int32, np.float64],
        'shapes': [(3,), (2, 2)],
        'names': ['ids', 'data_matrix'],
        'shared_name': '',
        'name': 'queue_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using a shared_name for inter-process communication.
    input_dict_3 = {
        'capacity': 50,
        'dtypes': [np.int64],
        'shapes': [(10,)],
        'names': ['shared_vector'],
        'shared_name': 'my_shared_queue_A',
        'name': 'queue_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Queue for boolean and uint8 types, common for images/masks.
    input_dict_4 = {
        'capacity': 20,
        'dtypes': [np.bool_, np.uint8],
        'shapes': [(256, 256), (256, 256, 3)],
        'names': ['mask', 'image'],
        'shared_name': '',
        'name': 'queue_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: High-dimensional tensor shape.
    input_dict_5 = {
        'capacity': 5,
        'dtypes': [np.float16],
        'shapes': [(4, 8, 8, 16)],
        'names': ['feature_tensor'],
        'shared_name': '',
        'name': 'queue_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: A queue for string data with a non-scalar shape.
    input_dict_6 = {
        'capacity': 15,
        'dtypes': [np.string_],
        'shapes': [(1,)],
        'names': ['message'],
        'shared_name': '',
        'name': 'queue_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: A queue for complex numbers.
    input_dict_7 = {
        'capacity': 30,
        'dtypes': [np.complex64],
        'shapes': [(128, 128)],
        'names': ['fft_output'],
        'shared_name': '',
        'name': 'queue_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Three components with varying non-scalar shapes.
    input_dict_8 = {
        'capacity': 40,
        'dtypes': [np.int32, np.float32, np.float32],
        'shapes': [(1,), (128,), (64, 64)],
        'names': ['id', 'embedding', 'context'],
        'shared_name': 'multi_component_queue_B',
        'name': 'queue_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Minimum capacity of 1.
    input_dict_9 = {
        'capacity': 1,
        'dtypes': [np.int8],
        'shapes': [(1,)],
        'names': ['single_byte'],
        'shared_name': '',
        'name': 'queue_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Large capacity for buffering.
    input_dict_10 = {
        'capacity': 5000,
        'dtypes': [np.uint64],
        'shapes': [(1024,)],
        'names': ['large_data_vector'],
        'shared_name': '',
        'name': 'queue_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.queue.FIFOQueue"] = tf_queue_FIFOQueue_inputs()

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
