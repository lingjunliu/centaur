
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_service_register_dataset_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid input
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    input_dict = {
        "service": "grpc://localhost:5000",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "dataset_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different service address
    dataset = tf.data.Dataset.from_tensor_slices(np.array([4, 5, 6]))
    input_dict = {
        "service": "grpc://127.0.0.1:6000",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "dataset_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: No compression
    dataset = tf.data.Dataset.from_tensor_slices(np.array([7, 8, 9]))
    input_dict = {
        "service": "grpc://localhost:5000",
        "dataset": dataset,
        "compression": None,
        "dataset_id": "dataset_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dataset ID
    dataset = tf.data.Dataset.from_tensor_slices(np.array([10, 11, 12]))
    input_dict = {
        "service": "grpc://localhost:5000",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "another_dataset"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: More complex dataset
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4], [5, 6]]))
    input_dict = {
        "service": "grpc://localhost:5000",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "complex_dataset"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Dataset with string elements
    dataset = tf.data.Dataset.from_tensor_slices(np.array(["a", "b", "c"]))
    input_dict = {
        "service": "grpc://localhost:5000",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "string_dataset"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Dataset with different data type
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.1, 2.2, 3.3]))
    input_dict = {
        "service": "grpc://localhost:5000",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "float_dataset"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Long dataset ID
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    input_dict = {
        "service": "grpc://localhost:5000",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "this_is_a_very_long_dataset_id_that_should_still_work"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.service.register_dataset"] = tf_data_experimental_service_register_dataset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.service.register_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.register_dataset'.")

check_valid('tf.data.experimental.service.register_dataset', generated_inputs['tf.data.experimental.service.register_dataset'], lib="tf", suffix=0)
