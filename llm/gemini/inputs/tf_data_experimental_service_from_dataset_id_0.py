
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_service_from_dataset_id_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.service.from_dataset_id.
    """
    list_of_inputs = []

    # Input 1: Simple service tuple
    input_1 = {
        'processing_mode': 'parallel_epochs',
        'service': ('grpc', 'localhost'),
        'dataset_id': 1,
        'element_spec': [tf.TensorSpec(shape=(), dtype=np.int64)],
        'job_name': 'job_1',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': -1,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: Local protocol
    input_2 = {
        'processing_mode': 'distributed_epoch',
        'service': ('local', 'worker'),
        'dataset_id': 2,
        'element_spec': [
            tf.TensorSpec(shape=(10,), dtype=np.float32),
            tf.TensorSpec(shape=(4, 2), dtype=np.int32)
        ],
        'job_name': 'job_2',
        'consumer_index': 1,
        'num_consumers': 2,
        'max_outstanding_requests': 100,
        'data_transfer_protocol': 'local',
        'cross_trainer_cache': [],
        'target_workers': 'LOCAL'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Empty string tuple for service
    input_3 = {
        'processing_mode': 'OFF',
        'service': ('', ''),
        'dataset_id': 123456789,
        'element_spec': [tf.TensorSpec(shape=(3, 3, 3), dtype=np.bool_)],
        'job_name': 'shared_job_off',
        'consumer_index': 0,
        'num_consumers': 4,
        'max_outstanding_requests': 0,
        'data_transfer_protocol': 'arrow',
        'cross_trainer_cache': [],
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: DYNAMIC sharding with IP address
    input_4 = {
        'processing_mode': 'DYNAMIC',
        'service': ('grpc', '127.0.0.1'),
        'dataset_id': 100,
        'element_spec': [tf.TensorSpec(shape=(5, 5), dtype=np.float64)],
        'job_name': 'dynamic_job',
        'consumer_index': 3,
        'num_consumers': 4,
        'max_outstanding_requests': 50,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: FILE sharding policy
    input_5 = {
        'processing_mode': 'FILE',
        'service': ('grpc', 'service'),
        'dataset_id': 99,
        'element_spec': [tf.TensorSpec(shape=(1,), dtype=np.int8)],
        'job_name': 'file_sharding_job',
        'consumer_index': 0,
        'num_consumers': 10,
        'max_outstanding_requests': 1000,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: DATA sharding with IPv6
    input_6 = {
        'processing_mode': 'DATA',
        'service': ('grpc', '::1'),
        'dataset_id': 500,
        'element_spec': [
            tf.TensorSpec(shape=(128, 128, 3), dtype=np.uint8),
            tf.TensorSpec(shape=(), dtype=np.int32)
        ],
        'job_name': 'image_pipeline_job',
        'consumer_index': 7,
        'num_consumers': 8,
        'max_outstanding_requests': 16,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'LOCAL'
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: FILE_OR_DATA sharding
    input_7 = {
        'processing_mode': 'FILE_OR_DATA',
        'service': ('grpc', '192.168.1'),
        'dataset_id': 2023,
        'element_spec': [tf.TensorSpec(shape=(10,), dtype=np.uint16)],
        'job_name': 'flexible_sharding_job',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 1,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: HINT sharding
    input_8 = {
        'processing_mode': 'HINT',
        'service': ('grpc', '0.0.0.0'),
        'dataset_id': 101,
        'element_spec': [tf.TensorSpec(shape=(4, 5, 6), dtype=np.float16)],
        'job_name': 'hint_sharding_job',
        'consumer_index': 4,
        'num_consumers': 5,
        'max_outstanding_requests': 20,
        'data_transfer_protocol': 'arrow',
        'cross_trainer_cache': [],
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: Large numbers for IDs and requests
    input_9 = {
        'processing_mode': 'OFF',
        'service': ('grpc', 'cluster'),
        'dataset_id': 987654321,
        'element_spec': [tf.TensorSpec(shape=(1024,), dtype=np.int32)],
        'job_name': 'large_scale_job',
        'consumer_index': 99,
        'num_consumers': 100,
        'max_outstanding_requests': 10000,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'LOCAL'
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: Empty element spec
    input_10 = {
        'processing_mode': 'parallel_epochs',
        'service': ('local', 'proc'),
        'dataset_id': 0,
        'element_spec': [],
        'job_name': 'minimal_job',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 1,
        'data_transfer_protocol': 'local',
        'cross_trainer_cache': [],
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_10))

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
