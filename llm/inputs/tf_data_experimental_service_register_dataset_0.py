
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_service_register_dataset_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    service = "grpc://localhost:5000"
    dataset = tf.constant(np.array([1, 2, 3]))
    compression = "AUTO"
    dataset_id = "dataset_1"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(copy.copy(input_dict))

    # Input 2: Different service address
    service = "grpc://127.0.0.1:6000"
    dataset = tf.constant(np.array([4, 5, 6, 7]))
    compression = None
    dataset_id = "dataset_2"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(copy.copy(input_dict))

    # Input 3: Different dataset with multiple dimensions
    service = "grpc://localhost:5000"
    dataset = tf.constant(np.array([[1, 2], [3, 4], [5, 6]]))
    compression = "AUTO"
    dataset_id = "dataset_3"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(copy.copy(input_dict))

    # Input 4: Empty dataset_id
    service = "grpc://localhost:5000"
    dataset = tf.constant(np.array([10, 11, 12]))
    compression = None
    dataset_id = ""
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(copy.copy(input_dict))

    # Input 5: Different compression algorithm
    service = "grpc://localhost:5000"
    dataset = tf.constant(np.array([13, 14, 15, 16, 17]))
    compression = "ZLIB" #Not really used according to documentation, so just put anything that is a string.
    dataset_id = "dataset_5"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(copy.copy(input_dict))

    # Input 6: Dataset with different data types
    service = "grpc://localhost:5000"
    dataset = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    compression = "AUTO"
    dataset_id = "dataset_6"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(copy.copy(input_dict))

    # Input 7: Dataset with string data
    service = "grpc://localhost:5000"
    dataset = tf.constant(np.array(["a", "b", "c"]))
    compression = None
    dataset_id = "dataset_7"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(copy.copy(input_dict))

    # Input 8: Longer dataset_id
    service = "grpc://localhost:5000"
    dataset = tf.constant(np.array([20, 21, 22, 23, 24, 25]))
    compression = "AUTO"
    dataset_id = "a_very_long_dataset_id_string"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(copy.copy(input_dict))

    # Input 9: Dataset with more dimensions
    service = "grpc://localhost:5000"
    dataset = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    compression = None
    dataset_id = "dataset_9"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(copy.copy(input_dict))

    # Input 10: Complex data type dataset
    service = "grpc://localhost:5000"
    dataset = tf.constant(np.array([1+1j, 2+2j, 3+3j]))
    compression = "AUTO"
    dataset_id = "dataset_10"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(copy.copy(input_dict))

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
