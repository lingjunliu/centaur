
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_service_register_dataset_inputs():
    list_of_inputs = []

    # Input 1: Basic dataset registration
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    input_dict = {
        "service": "grpc://localhost:5000",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "dataset_1"
    }
    list_of_inputs.append(input_dict)

    # Input 2: No compression
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.map(lambda x: tf.cast(x, tf.int64))
    dataset = tf.data.Dataset.from_tensor_slices(np.array(list(dataset.as_numpy_iterator())))
    input_dict = {
        "service": "grpc://localhost:5001",
        "dataset": dataset,
        "compression": None,
        "dataset_id": "dataset_2"
    }
    list_of_inputs.append(input_dict)

    # Input 3: Different dataset type (string)
    dataset = tf.data.Dataset.from_tensor_slices(np.array(["a", "b", "c"]))
    input_dict = {
        "service": "grpc://localhost:5002",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "dataset_3"
    }
    list_of_inputs.append(input_dict)

    # Input 4: Dataset with multiple components
    feature = np.array([1, 2])
    label = np.array([0, 1])
    dataset = tf.data.Dataset.from_tensor_slices((feature, label))
    input_dict = {
        "service": "grpc://localhost:5003",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "dataset_4"
    }
    list_of_inputs.append(input_dict)

    # Input 5: Different service address
    dataset = tf.data.Dataset.from_tensor_slices(np.array([4, 5, 6]))
    input_dict = {
        "service": "grpc://127.0.0.1:5004",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "dataset_5"
    }
    list_of_inputs.append(input_dict)

    # Input 6: Empty dataset
    dataset = tf.data.Dataset.from_tensor_slices(np.array([]))
    input_dict = {
        "service": "grpc://localhost:5005",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "dataset_6"
    }
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with floats
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0]))
    input_dict = {
        "service": "grpc://localhost:5006",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "dataset_7"
    }
    list_of_inputs.append(input_dict)

    # Input 8: Longer dataset_id
    dataset = tf.data.Dataset.from_tensor_slices(np.array([7, 8, 9]))
    input_dict = {
        "service": "grpc://localhost:5007",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "very_long_dataset_id_1234567890"
    }
    list_of_inputs.append(input_dict)

    # Input 9: Tuple service
    dataset = tf.data.Dataset.from_tensor_slices(np.array([10, 11, 12]))
    input_dict = {
        "service": "localhost:5008",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "dataset_9"
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: Multiple dimensions
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4], [5, 6]]))
    input_dict = {
        "service": "grpc://localhost:5009",
        "dataset": dataset,
        "compression": "AUTO",
        "dataset_id": "dataset_10"
    }
    list_of_inputs.append(input_dict)

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
