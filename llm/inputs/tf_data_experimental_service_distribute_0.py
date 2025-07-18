
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy


def tf_data_experimental_service_distribute_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.service.distribute.
    """
    list_of_inputs = []
    dataset_1d = np.arange(20, dtype=np.int64)
    dataset_2d = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)

    # Input 1: Basic 'distributed_epoch' mode
    input_1 = {
        'self': dataset_1d,
        'processing_mode': 'distributed_epoch',
        'service': 'grpc://dispatcher.example.com:5000',
        'job_name': 'job_1',
        'consumer_index': np.int64(0),
        'num_consumers': np.int64(1),
        'max_outstanding_requests': np.int64(10),
        'data_transfer_protocol': 'grpc',
        'compression': 'AUTO',
        'cross_trainer_cache': (),
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: Basic 'parallel_epochs' mode with 2D dataset
    input_2 = {
        'self': dataset_2d,
        'processing_mode': 'parallel_epochs',
        'service': '127.0.0.1:10000',
        'job_name': 'job_2',
        'consumer_index': np.int64(0),
        'num_consumers': np.int64(2),
        'max_outstanding_requests': np.int64(20),
        'data_transfer_protocol': 'grpc',
        'compression': 'AUTO',
        'cross_trainer_cache': (),
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Coordinated Read with multiple consumers
    input_3 = {
        'self': dataset_1d,
        'processing_mode': 'parallel_epochs',
        'service': 'tf-data-service:9090',
        'job_name': 'coord_read_job',
        'consumer_index': np.int64(0),
        'num_consumers': np.int64(4),
        'max_outstanding_requests': np.int64(100),
        'data_transfer_protocol': 'grpc',
        'compression': 'AUTO',
        'cross_trainer_cache': (),
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Another consumer for the same coordinated read job
    input_4 = {
        'self': dataset_1d,
        'processing_mode': 'parallel_epochs',
        'service': 'tf-data-service:9090',
        'job_name': 'coord_read_job',
        'consumer_index': np.int64(3),
        'num_consumers': np.int64(4),
        'max_outstanding_requests': np.int64(100),
        'data_transfer_protocol': 'grpc',
        'compression': 'SNAPPY',
        'cross_trainer_cache': (),
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: With SNAPPY compression
    input_5 = {
        'self': dataset_1d,
        'processing_mode': 'parallel_epochs',
        'service': 'dispatcher:2222',
        'job_name': 'job_5',
        'consumer_index': np.int64(1),
        'num_consumers': np.int64(2),
        'max_outstanding_requests': np.int64(1),
        'data_transfer_protocol': 'grpc',
        'compression': 'SNAPPY',
        'cross_trainer_cache': (),
        'target_workers': 'LOCAL'
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: With target_workers set to LOCAL
    input_6 = {
        'self': dataset_2d,
        'processing_mode': 'distributed_epoch',
        'service': 'localhost:1111',
        'job_name': 'job_6',
        'consumer_index': np.int64(0),
        'num_consumers': np.int64(1),
        'max_outstanding_requests': np.int64(50),
        'data_transfer_protocol': 'grpc',
        'compression': 'AUTO',
        'cross_trainer_cache': (),
        'target_workers': 'LOCAL'
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Varied parameters
    input_7 = {
        'self': dataset_1d,
        'processing_mode': 'distributed_epoch',
        'service': 'localhost:1112',
        'job_name': 'job_7_snappy',
        'consumer_index': np.int64(1),
        'num_consumers': np.int64(2),
        'max_outstanding_requests': np.int64(500),
        'data_transfer_protocol': 'grpc',
        'compression': 'SNAPPY',
        'cross_trainer_cache': (),
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: Target workers set to ANY
    input_8 = {
        'self': dataset_1d,
        'processing_mode': 'parallel_epochs',
        'service': 'any-worker-service:8888',
        'job_name': 'job_8',
        'consumer_index': np.int64(7),
        'num_consumers': np.int64(8),
        'max_outstanding_requests': np.int64(80),
        'data_transfer_protocol': 'grpc',
        'compression': 'AUTO',
        'cross_trainer_cache': (),
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: Small values for integer parameters
    input_9 = {
        'self': dataset_2d,
        'processing_mode': 'distributed_epoch',
        'service': '192.168.1.100:5000',
        'job_name': 'job_9',
        'consumer_index': np.int64(0),
        'num_consumers': np.int64(1),
        'max_outstanding_requests': np.int64(1),
        'data_transfer_protocol': 'grpc',
        'compression': 'SNAPPY',
        'cross_trainer_cache': (),
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: Large values for integer parameters
    input_10 = {
        'self': np.arange(1000, dtype=np.int64),
        'processing_mode': 'parallel_epochs',
        'service': 'large-cluster-dispatcher:5000',
        'job_name': 'job_10_large',
        'consumer_index': np.int64(99),
        'num_consumers': np.int64(100),
        'max_outstanding_requests': np.int64(1000),
        'data_transfer_protocol': 'grpc',
        'compression': 'AUTO',
        'cross_trainer_cache': (),
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs


generated_inputs["tf.data.experimental.service.distribute"] = tf_data_experimental_service_distribute_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.service.distribute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.distribute'.")

check_valid('tf.data.experimental.service.distribute', generated_inputs['tf.data.experimental.service.distribute'], lib="tf", suffix=0)
