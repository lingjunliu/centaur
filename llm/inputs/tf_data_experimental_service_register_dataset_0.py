
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_service_register_dataset_inputs():
    list_of_inputs = []

    # Input 1
    service = np.array("grpc://localhost:50051", dtype=np.str_)
    dataset = tf.constant(np.array([1, 2, 3], dtype=np.int64))
    compression = np.array("AUTO", dtype=np.str_)
    dataset_id = np.array("dataset_1", dtype=np.str_)
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict)

    # Input 2
    service = np.array("grpc://localhost:50052", dtype=np.str_)
    dataset = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int64))
    compression = np.array("", dtype=np.str_)
    dataset_id = np.array("dataset_2", dtype=np.str_)
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict)

    # Input 3
    service = np.array("grpc://localhost:50053", dtype=np.str_)
    dataset = tf.constant(np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.int64))
    compression = np.array("AUTO", dtype=np.str_)
    dataset_id = np.array("dataset_3", dtype=np.str_)
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict)

    # Input 4
    service = np.array("grpc://localhost:50054", dtype=np.str_)
    dataset = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    compression = np.array("", dtype=np.str_)
    dataset_id = np.array("dataset_4", dtype=np.str_)
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
    list_of_inputs.append(input_dict)

    # Input 5
    service = np.array("grpc://localhost:50055", dtype=np.str_)
    dataset = tf.constant(np.array(["a", "b", "c"], dtype=np.str_))
    compression = np.array("AUTO", dtype=np.str_)
    dataset_id = np.array("dataset_5", dtype=np.str_)
    input_dict = {"service": service, "dataset": dataset, "compression": compression, "dataset_id": dataset_id}
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
