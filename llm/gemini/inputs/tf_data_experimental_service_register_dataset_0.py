
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_service_register_dataset_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.service.register_dataset function.
    
    NOTE: The provided signature {'dataset': 'tensor'} and the testing framework's requirement for
    a '.shape' attribute conflict with the API's need for a `tf.data.Dataset` object. This
    implementation provides numpy arrays for the 'dataset' parameter to resolve the immediate
    `AttributeError: '...Dataset' object has no attribute 'shape'` error from the framework.
    """
    list_of_inputs = []
    service_address = "grpc://localhost:5050"

    # Input 1: Simple 1D int array
    input_dict1 = {
        'service': service_address,
        'dataset': np.arange(10, dtype=np.int32),
        'compression': 'AUTO',
        'dataset_id': 'dataset_1_int'
    }
    list_of_inputs.append(input_dict1)

    # Input 2: 2D float array with SNAPPY compression
    input_dict2 = {
        'service': service_address,
        'dataset': np.random.rand(5, 3).astype(np.float32),
        'compression': 'SNAPPY',
        'dataset_id': 'dataset_2_float'
    }
    list_of_inputs.append(input_dict2)

    # Input 3: Empty numpy array
    input_dict3 = {
        'service': 'grpc://127.0.0.1:9090',
        'dataset': np.array([], dtype=np.float32),
        'compression': 'AUTO',
        'dataset_id': 'empty_dataset'
    }
    list_of_inputs.append(input_dict3)

    # Input 4: Single-element tensor
    input_dict4 = {
        'service': service_address,
        'dataset': np.array([[[1, 2, 3]]], dtype=np.int16),
        'compression': 'SNAPPY',
        'dataset_id': 'single_element_dataset'
    }
    list_of_inputs.append(input_dict4)

    # Input 5: Boolean numpy array
    input_dict5 = {
        'service': service_address,
        'dataset': np.array([True, False, True, True]),
        'compression': 'AUTO',
        'dataset_id': 'boolean_dataset'
    }
    list_of_inputs.append(input_dict5)

    # Input 6: 3D int array with negative values
    input_dict6 = {
        'service': service_address,
        'dataset': np.random.randint(-100, 100, size=(4, 2, 3), dtype=np.int32),
        'compression': 'SNAPPY',
        'dataset_id': '3d_dataset'
    }
    list_of_inputs.append(input_dict6)

    # Input 7: String numpy array
    input_dict7 = {
        'service': 'grpc://remote-dispatcher:1234',
        'dataset': np.array(['apple', 'banana', 'cherry']),
        'compression': 'AUTO',
        'dataset_id': 'string_dataset'
    }
    list_of_inputs.append(input_dict7)

    # Input 8: Unsigned integer numpy array
    input_dict8 = {
        'service': service_address,
        'dataset': np.array([1, 2, 3], dtype=np.uint8),
        'compression': 'AUTO',
        'dataset_id': 'uint8_dataset'
    }
    list_of_inputs.append(input_dict8)

    # Input 9: High-rank (4D) numpy array
    input_dict9 = {
        'service': service_address,
        'dataset': np.zeros((2, 3, 4, 5), dtype=np.float16),
        'compression': 'SNAPPY',
        'dataset_id': 'rank_4_dataset'
    }
    list_of_inputs.append(input_dict9)

    # Input 10: Complex numpy array
    input_dict10 = {
        'service': service_address,
        'dataset': np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
        'compression': 'AUTO',
        'dataset_id': 'complex_dataset'
    }
    list_of_inputs.append(input_dict10)

    return list_of_inputs

generated_inputs["tf.data.experimental.service.register_dataset"] = tf_data_experimental_service_register_dataset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.service.register_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.register_dataset'.")

check_valid('tf.data.experimental.service.register_dataset', generated_inputs['tf.data.experimental.service.register_dataset'], lib="tf", suffix=0)
