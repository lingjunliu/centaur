
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_service_register_dataset_inputs():
    list_of_inputs = []

    # Input 1
    service = "grpc://localhost:5000"
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    compression = "AUTO"
    dataset_id = "dataset1"

    input_dict = {
        "service": service,
        "dataset": dataset,
        "compression": compression,
        "dataset_id": dataset_id
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    service = "grpc://localhost:5001"
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4]]))
    compression = None
    dataset_id = "dataset2"

    input_dict = {
        "service": service,
        "dataset": dataset,
        "compression": compression,
        "dataset_id": dataset_id
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    service = "grpc://localhost:5002"
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    compression = "AUTO"
    dataset_id = "dataset3"

    input_dict = {
        "service": service,
        "dataset": dataset,
        "compression": compression,
        "dataset_id": dataset_id
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    service = "grpc://localhost:5003"
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0]))
    compression = None
    dataset_id = "dataset4"

    input_dict = {
        "service": service,
        "dataset": dataset,
        "compression": compression,
        "dataset_id": dataset_id
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    service = "grpc://localhost:5004"
    dataset = tf.data.Dataset.from_tensor_slices(np.array(["a", "b", "c"]))
    compression = "AUTO"
    dataset_id = "dataset5"

    input_dict = {
        "service": service,
        "dataset": dataset,
        "compression": compression,
        "dataset_id": dataset_id
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    service = "grpc://localhost:5005"
    dataset = tf.data.Dataset.from_tensor_slices(np.array([True, False, True]))
    compression = None
    dataset_id = "dataset6"

    input_dict = {
        "service": service,
        "dataset": dataset,
        "compression": compression,
        "dataset_id": dataset_id
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    service = "grpc://localhost:5006"
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    compression = "AUTO"
    dataset_id = "dataset7"

    input_dict = {
        "service": service,
        "dataset": dataset,
        "compression": compression,
        "dataset_id": dataset_id
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    service = "grpc://localhost:5007"
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.1, 2.2, 3.3], dtype=np.float64))
    compression = None
    dataset_id = "dataset8"

    input_dict = {
        "service": service,
        "dataset": dataset,
        "compression": compression,
        "dataset_id": dataset_id
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    service = "grpc://localhost:5008"
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2]))
    compression = "AUTO"
    dataset_id = ""

    input_dict = {
        "service": service,
        "dataset": dataset,
        "compression": compression,
        "dataset_id": dataset_id
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    service = "grpc://localhost:5009"
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1]))
    compression = None
    dataset_id = "dataset10"

    input_dict = {
        "service": service,
        "dataset": dataset,
        "compression": compression,
        "dataset_id": dataset_id
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
