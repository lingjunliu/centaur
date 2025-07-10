
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_service_from_dataset_id_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'processing_mode': 'parallel_epochs',
        'service': ('grpc', 'localhost:5000'),
        'dataset_id': 123,
        'element_spec': [tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None)],
        'job_name': 'job1',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 10,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'processing_mode': 'distributed_epoch',
        'service': ('grpc', '192.168.1.10:8000'),
        'dataset_id': 456,
        'element_spec': [tf.TensorSpec(shape=(28, 28), dtype=tf.float32, name=None)],
        'job_name': 'job2',
        'consumer_index': 1,
        'num_consumers': 4,
        'max_outstanding_requests': 20,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'processing_mode': 'parallel_epochs',
        'service': ('grpc', 'remote_host:9000'),
        'dataset_id': 789,
        'element_spec': [tf.TensorSpec(shape=(), dtype=tf.string, name=None)],
        'job_name': 'job3',
        'consumer_index': None,
        'num_consumers': None,
        'max_outstanding_requests': 30,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'LOCAL'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'processing_mode': 'distributed_epoch',
        'service': ('grpc', '10.0.0.5:7000'),
        'dataset_id': 101,
        'element_spec': [tf.TensorSpec(shape=(None, 10), dtype=tf.int32, name=None)],
        'job_name': 'job4',
        'consumer_index': None,
        'num_consumers': None,
        'max_outstanding_requests': 40,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'processing_mode': 'parallel_epochs',
        'service': ('grpc', 'localhost:6000'),
        'dataset_id': 202,
        'element_spec': [tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8, name=None)],
        'job_name': 'job5',
        'consumer_index': None,
        'num_consumers': None,
        'max_outstanding_requests': 50,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'processing_mode': 'distributed_epoch',
        'service': ('grpc', '172.17.0.1:4000'),
        'dataset_id': 303,
        'element_spec': [tf.TensorSpec(shape=(1,), dtype=tf.float64, name=None)],
        'job_name': 'job6',
        'consumer_index': None,
        'num_consumers': None,
        'max_outstanding_requests': 60,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'LOCAL'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_dict = {
        'processing_mode': 'parallel_epochs',
        'service': ('grpc', '127.0.0.1:5001'),
        'dataset_id': 404,
        'element_spec': [tf.TensorSpec(shape=(None, None), dtype=tf.string, name=None)],
        'job_name': 'job7',
        'consumer_index': None,
        'num_consumers': None,
        'max_outstanding_requests': 70,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'processing_mode': 'distributed_epoch',
        'service': ('grpc', 'localhost:8001'),
        'dataset_id': 505,
        'element_spec': [tf.TensorSpec(shape=(128,), dtype=tf.int8, name=None)],
        'job_name': 'job8',
        'consumer_index': None,
        'num_consumers': None,
        'max_outstanding_requests': 80,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'processing_mode': 'parallel_epochs',
        'service': ('grpc', '::1:9001'),
        'dataset_id': 606,
        'element_spec': [tf.TensorSpec(shape=(2,2,2), dtype=tf.bool, name=None)],
        'job_name': 'job9',
        'consumer_index': None,
        'num_consumers': None,
        'max_outstanding_requests': 90,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'LOCAL'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'processing_mode': 'distributed_epoch',
        'service': ('grpc', '[2001:db8::1]:7001'),
        'dataset_id': 707,
        'element_spec': [tf.TensorSpec(shape=(None, 784), dtype=tf.bfloat16, name=None)],
        'job_name': 'job10',
        'consumer_index': None,
        'num_consumers': None,
        'max_outstanding_requests': 100,
        'data_transfer_protocol': 'grpc',
        'cross_trainer_cache': [],
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.service.from_dataset_id"] = tf_data_experimental_service_from_dataset_id_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.service.from_dataset_id' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.from_dataset_id'.")

check_valid('tf.data.experimental.service.from_dataset_id', generated_inputs['tf.data.experimental.service.from_dataset_id'], lib="tf", suffix=0)
