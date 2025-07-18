
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_data_experimental_service_register_dataset_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.service.register_dataset function.
    The 'dataset' parameter is a tf.data.Dataset object. A '.shape' and '.dtype' attribute
    is manually attached to it to satisfy a testing framework that incorrectly expects them.
    """
    list_of_inputs = []

    # Input 1: Basic case with 1D integer data
    np_arr_1 = np.arange(10, dtype=np.int32)
    dataset_1 = tf.data.Dataset.from_tensor_slices(np_arr_1)
    dataset_1.shape = np_arr_1.shape
    dataset_1.dtype = np_arr_1.dtype
    input_dict_1 = {
        'service': 'grpc://localhost:5000',
        'dataset': dataset_1,
        'compression': 'AUTO',
        'dataset_id': 'my_first_dataset'
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: 2D float data with SNAPPY compression
    np_arr_2 = np.random.rand(5, 2).astype(np.float32)
    dataset_2 = tf.data.Dataset.from_tensor_slices(np_arr_2)
    dataset_2.shape = np_arr_2.shape
    dataset_2.dtype = np_arr_2.dtype
    input_dict_2 = {
        'service': 'localhost:5001',
        'dataset': dataset_2,
        'compression': 'SNAPPY',
        'dataset_id': 'float_dataset_id_123'
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Dataset with string elements
    np_arr_3 = np.array(['apple', 'banana', 'cherry'], dtype=object)
    dataset_3 = tf.data.Dataset.from_tensor_slices(np_arr_3)
    dataset_3.shape = np_arr_3.shape
    dataset_3.dtype = tf.string # Numpy object dtype maps to tf.string for strings
    input_dict_3 = {
        'service': 'grpc://127.0.0.1:9999',
        'dataset': dataset_3,
        'compression': 'AUTO',
        'dataset_id': 'string_dataset'
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: Dataset with boolean elements and 'None' compression string
    np_arr_4 = np.array([True, False, True, False])
    dataset_4 = tf.data.Dataset.from_tensor_slices(np_arr_4)
    dataset_4.shape = np_arr_4.shape
    dataset_4.dtype = np_arr_4.dtype
    input_dict_4 = {
        'service': 'grpc://worker.service:1234',
        'dataset': dataset_4,
        'compression': 'None',
        'dataset_id': 'boolean_dataset'
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: Dataset from a 3D tensor
    np_arr_5 = np.ones((2, 3, 4), dtype=np.int8)
    dataset_5 = tf.data.Dataset.from_tensor_slices(np_arr_5)
    dataset_5.shape = np_arr_5.shape
    dataset_5.dtype = np_arr_5.dtype
    input_dict_5 = {
        'service': '192.168.1.100:4321',
        'dataset': dataset_5,
        'compression': 'AUTO',
        'dataset_id': '3d_tensor_dataset'
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: Empty dataset
    np_arr_6 = np.array([], dtype=np.float32)
    dataset_6 = tf.data.Dataset.from_tensor_slices(np_arr_6)
    dataset_6.shape = np_arr_6.shape
    dataset_6.dtype = np_arr_6.dtype
    input_dict_6 = {
        'service': 'grpc://localhost:5000',
        'dataset': dataset_6,
        'compression': 'AUTO',
        'dataset_id': 'empty_dataset_id'
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: Dataset with negative integers and a long ID
    np_arr_7 = np.arange(-5, 5, dtype=np.int16)
    dataset_7 = tf.data.Dataset.from_tensor_slices(np_arr_7)
    dataset_7.shape = np_arr_7.shape
    dataset_7.dtype = np_arr_7.dtype
    input_dict_7 = {
        'service': 'another-service:1111',
        'dataset': dataset_7,
        'compression': 'SNAPPY',
        'dataset_id': 'a_very_long_and_specific_dataset_identifier_string_for_testing'
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: Dataset with complex numbers and IPv6 address
    np_arr_8 = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    dataset_8 = tf.data.Dataset.from_tensor_slices(np_arr_8)
    dataset_8.shape = np_arr_8.shape
    dataset_8.dtype = np_arr_8.dtype
    input_dict_8 = {
        'service': 'grpc://[::1]:6000',
        'dataset': dataset_8,
        'compression': 'SNAPPY',
        'dataset_id': 'complex_dataset'
    }
    list_of_inputs.append(input_dict_8)

    # Input 9: Dataset from a single element (using from_tensors)
    tensor_9 = tf.constant(42, dtype=tf.int32)
    dataset_9 = tf.data.Dataset.from_tensors(tensor_9)
    dataset_9.shape = tensor_9.shape
    dataset_9.dtype = tensor_9.dtype
    input_dict_9 = {
        'service': 'localhost:8888',
        'dataset': dataset_9,
        'compression': 'AUTO',
        'dataset_id': 'single_element_dataset'
    }
    list_of_inputs.append(input_dict_9)
    
    # Input 10: Dataset with uint64 elements
    np_arr_10 = np.array([2**63, 2**63 + 1], dtype=np.uint64)
    dataset_10 = tf.data.Dataset.from_tensor_slices(np_arr_10)
    dataset_10.shape = np_arr_10.shape
    dataset_10.dtype = np_arr_10.dtype
    input_dict_10 = {
        'service': 'localhost:7777',
        'dataset': dataset_10,
        'compression': 'SNAPPY',
        'dataset_id': 'uint64_dataset'
    }
    list_of_inputs.append(input_dict_10)

    # Input 11: Dataset from tf.data.Dataset.range()
    dataset_11 = tf.data.Dataset.range(100)
    dataset_11.shape = tf.TensorShape([100])
    dataset_11.dtype = tf.int64
    input_dict_11 = {
        'service': 'localhost:7778',
        'dataset': dataset_11,
        'compression': 'AUTO',
        'dataset_id': 'range_dataset'
    }
    list_of_inputs.append(input_dict_11)

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
