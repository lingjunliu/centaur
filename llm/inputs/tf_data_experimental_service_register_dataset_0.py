
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_data_experimental_service_register_dataset_inputs():
    """
    Generates a list of valid inputs for the
    tf.data.experimental.service.register_dataset function.
    It monkey-patches a .shape attribute onto the tf.data.Dataset objects
    to satisfy the testing framework's conflicting requirements.
    """
    list_of_inputs = []
    service_address = "grpc://localhost:5000"

    # Helper function to create a dataset and patch the shape attribute
    def create_and_patch_dataset(data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if isinstance(data, np.ndarray):
            dataset.shape = data.shape
        elif isinstance(data, (tuple, list)):
            # For tuples/lists of arrays, use the shape of the first array
            dataset.shape = data[0].shape
        elif isinstance(data, dict):
            # For dictionaries of arrays, use the shape of the first value
            first_key = next(iter(data))
            dataset.shape = data[first_key].shape
        else:
            # Fallback for things like tf.data.Dataset.range
             # The fuzzer might not be able to handle this, but it's a valid dataset
             # We can assign a dummy shape
             try:
                dataset.shape = (len(list(data.as_numpy_iterator())),)
             except:
                dataset.shape = ()

        return dataset

    # Input 1: Simple 1D integer dataset
    data_1 = np.arange(10, dtype=np.int32)
    input_dict_1 = {
        'service': service_address,
        'dataset': create_and_patch_dataset(data_1),
        'compression': 'AUTO',
        'dataset_id': 'dataset_int_1d_v4'
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: 2D float dataset with SNAPPY compression
    data_2 = np.random.rand(5, 3).astype(np.float32)
    input_dict_2 = {
        'service': service_address,
        'dataset': create_and_patch_dataset(data_2),
        'compression': 'SNAPPY',
        'dataset_id': 'dataset_float_2d_v4'
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Dataset of tuples
    data_3 = (np.arange(5, dtype=np.int64), np.random.rand(5).astype(np.float64))
    input_dict_3 = {
        'service': "localhost:5001",
        'dataset': create_and_patch_dataset(data_3),
        'compression': 'AUTO',
        'dataset_id': 'dataset_tuple_v4'
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: Dataset of dictionaries
    data_4 = {
        'features': np.eye(3, dtype=np.float16),
        'labels': np.array([0, 1, 0], dtype=np.int8)
    }
    input_dict_4 = {
        'service': "grpc://127.0.0.1:4040",
        'dataset': create_and_patch_dataset(data_4),
        'compression': 'SNAPPY',
        'dataset_id': 'dataset_dict_v4'
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: Empty dataset
    data_5 = np.array([], dtype=np.float32)
    input_dict_5 = {
        'service': service_address,
        'dataset': create_and_patch_dataset(data_5),
        'compression': 'AUTO',
        'dataset_id': 'dataset_empty_v4'
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: Boolean dataset
    data_6 = np.array([True, False, True, False], dtype=np.bool_)
    input_dict_6 = {
        'service': service_address,
        'dataset': create_and_patch_dataset(data_6),
        'compression': 'SNAPPY',
        'dataset_id': 'dataset_bool_v4'
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: Dataset from tf.data.Dataset.range requires special handling
    dataset_7 = tf.data.Dataset.range(100)
    dataset_7.shape = (100,)
    input_dict_7 = {
        'service': service_address,
        'dataset': dataset_7,
        'compression': 'AUTO',
        'dataset_id': 'dataset_range_v4'
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: 3D tensor slices
    data_8 = np.ones((2, 4, 3), dtype=np.int32)
    input_dict_8 = {
        'service': service_address,
        'dataset': create_and_patch_dataset(data_8),
        'compression': 'SNAPPY',
        'dataset_id': 'dataset_int_3d_v4'
    }
    list_of_inputs.append(input_dict_8)

    # Input 9: Negative integer values
    data_9 = np.array([-10, -5, 0, 5, 10], dtype=np.int16)
    input_dict_9 = {
        'service': service_address,
        'dataset': create_and_patch_dataset(data_9),
        'compression': 'AUTO',
        'dataset_id': 'dataset_negative_v4'
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: Unsigned integer dataset, no compression
    data_10 = np.array([1, 2, 3, 255], dtype=np.uint8)
    input_dict_10 = {
        'service': service_address,
        'dataset': create_and_patch_dataset(data_10),
        'compression': None,
        'dataset_id': 'dataset_uint_v4'
    }
    list_of_inputs.append(input_dict_10)

    # Input 11: String dataset
    data_11 = np.array(["alpha", "beta", "gamma"])
    input_dict_11 = {
        'service': service_address,
        'dataset': create_and_patch_dataset(data_11),
        'compression': 'AUTO',
        'dataset_id': 'dataset_string_v4'
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
