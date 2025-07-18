
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_service_from_dataset_id_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.service.from_dataset_id function.
    """
    list_of_inputs = []

    # Input 1: Basic case with "OFF"
    input_dict_1 = {
        'processing_mode': 'OFF',
        'service': ('grpc', 'a'),
        'dataset_id': 1,
        'element_spec': [tf.TensorSpec(shape=(), dtype=np.int32)],
        'job_name': 'j1',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 10,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: "DYNAMIC" with multiple consumers
    input_dict_2 = {
        'processing_mode': 'DYNAMIC',
        'service': ('local', 'b'),
        'dataset_id': 2,
        'element_spec': [tf.TensorSpec(shape=(3,), dtype=np.float32)],
        'job_name': 'j2',
        'consumer_index': 1,
        'num_consumers': 2,
        'max_outstanding_requests': 100,
        'data_transfer_protocol': '',
        'cross_trainer_cache': [],
        'target_workers': "ANY"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: "FILE" sharding
    input_dict_3 = {
        'processing_mode': 'FILE',
        'service': ('grpc', 'c'),
        'dataset_id': 100,
        'element_spec': [tf.TensorSpec(shape=(2, 2), dtype=np.string_), tf.TensorSpec(shape=(), dtype=np.bool_)],
        'job_name': 'j3',
        'consumer_index': 0,
        'num_consumers': 4,
        'max_outstanding_requests': 5,
        'data_transfer_protocol': 'arrow',
        'cross_trainer_cache': [],
        'target_workers': "LOCAL"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: "DATA" sharding
    input_dict_4 = {
        'processing_mode': 'DATA',
        'service': ('grpc', 'd'),
        'dataset_id': 999,
        'element_spec': [tf.TensorSpec(shape=(None,), dtype=np.int64)],
        'job_name': 'j4',
        'consumer_index': 3,
        'num_consumers': 4,
        'max_outstanding_requests': 1,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Backwards compatibility "parallel_epochs"
    input_dict_5 = {
        'processing_mode': 'parallel_epochs',
        'service': ('grpc', 'e'),
        'dataset_id': 2147483647,
        'element_spec': [tf.TensorSpec(shape=(10, None, 3), dtype=np.float16)],
        'job_name': 'j5',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 20,
        'data_transfer_protocol': '',
        'cross_trainer_cache': [],
        'target_workers': "ANY"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Backwards compatibility "distributed_epoch"
    input_dict_6 = {
        'processing_mode': 'distributed_epoch',
        'service': ('grpc', 'f'),
        'dataset_id': 0,
        'element_spec': [tf.TensorSpec(shape=(), dtype=np.uint8)],
        'job_name': 'j6',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 0,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: "HINT" sharding
    input_dict_7 = {
        'processing_mode': 'HINT',
        'service': ('local', ''),
        'dataset_id': 77,
        'element_spec': [tf.TensorSpec(shape=(5,), dtype=np.int32), tf.TensorSpec(shape=(5, 5), dtype=np.float64), tf.TensorSpec(shape=(), dtype=np.string_)],
        'job_name': 'j7',
        'consumer_index': 2,
        'num_consumers': 3,
        'max_outstanding_requests': 50,
        'data_transfer_protocol': 'arrow',
        'cross_trainer_cache': [],
        'target_workers': "LOCAL"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: High number of consumers
    input_dict_8 = {
        'processing_mode': 'OFF',
        'service': ('grpc', 'h'),
        'dataset_id': 8,
        'element_spec': [tf.TensorSpec(shape=(None, 28, 28, 1), dtype=np.float32)],
        'job_name': 'j8',
        'consumer_index': 99,
        'num_consumers': 100,
        'max_outstanding_requests': 128,
        'data_transfer_protocol': '',
        'cross_trainer_cache': [],
        'target_workers': "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Empty protocol string in service
    input_dict_9 = {
        'processing_mode': 'DYNAMIC',
        'service': ('', 'i'),
        'dataset_id': 9001,
        'element_spec': [tf.TensorSpec(shape=(), dtype=np.complex64)],
        'job_name': 'j9',
        'consumer_index': 0,
        'num_consumers': 2,
        'max_outstanding_requests': 16,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': "ANY"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: bfloat16
    input_dict_10 = {
        'processing_mode': 'DATA',
        'service': ('grpc', 'k'),
        'dataset_id': 101010,
        'element_spec': [tf.TensorSpec(shape=(None, 10), dtype=tf.bfloat16.as_numpy_dtype)],
        'job_name': 'j10',
        'consumer_index': 7,
        'num_consumers': 8,
        'max_outstanding_requests': 32,
        'data_transfer_protocol': 'arrow',
        'cross_trainer_cache': [],
        'target_workers': "LOCAL"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.service.from_dataset_id"] = tf_data_experimental_service_from_dataset_id_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.service.from_dataset_id' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.from_dataset_id'.")

check_valid('tf.data.experimental.service.from_dataset_id', generated_inputs['tf.data.experimental.service.from_dataset_id'], lib="tf", suffix=0)
