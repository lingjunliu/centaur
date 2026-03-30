
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_service_register_dataset_inputs():
    list_of_inputs = []

    def create_dataset_from_numpy(numpy_array):
        return tf.data.Dataset.from_tensor_slices(numpy_array)

    # Input 1
    service = "grpc://localhost:5000"
    dataset = create_dataset_from_numpy(np.array([1, 2, 3]))
    compression = "AUTO"
    dataset_id = "dataset_1"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict.copy())

    # Input 2
    service = "grpc://localhost:5001"
    dataset = create_dataset_from_numpy(np.arange(100))
    compression = "SNAPPY"
    dataset_id = "dataset_2"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict.copy())

    # Input 3
    service = "grpc://localhost:5002"
    dataset = create_dataset_from_numpy(np.array([[1, 2], [3, 4]]))
    compression = None
    dataset_id = "dataset_3"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict.copy())

    # Input 4
    service = "grpc://localhost:5003"
    dataset = create_dataset_from_numpy(np.array([1.0, 2.0, 3.0]))
    compression = "AUTO"
    dataset_id = "dataset_4"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict.copy())

    # Input 5
    service = "grpc://localhost:5004"
    dataset = create_dataset_from_numpy(np.array(["a", "b", "c"]))
    compression = "SNAPPY"
    dataset_id = "dataset_5"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict.copy())

    # Input 6
    service = "grpc://localhost:5005"
    dataset = create_dataset_from_numpy(np.array([True, False, True]))
    compression = None
    dataset_id = "dataset_6"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict.copy())

    # Input 7
    service = "grpc://localhost:5006"
    dataset = create_dataset_from_numpy(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    compression = "AUTO"
    dataset_id = "dataset_7"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict.copy())

    # Input 8
    service = "grpc://localhost:5007"
    dataset = create_dataset_from_numpy(np.array([[1,2], [3,4]], dtype=np.int32))
    compression = "SNAPPY"
    dataset_id = "dataset_8"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict.copy())

    # Input 9
    service = "grpc://localhost:5008"
    dataset = create_dataset_from_numpy(np.array([1, 2, 3]).astype(np.int64))
    compression = None
    dataset_id = "dataset_9"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict.copy())

    # Input 10
    service = "grpc://localhost:5009"
    dataset = create_dataset_from_numpy(np.random.rand(10, 5))
    compression = "AUTO"
    dataset_id = "dataset_10"
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict.copy())

    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.service.register_dataset', generated_inputs['tf.data.experimental.service.register_dataset'], lib="tf", suffix=0)
