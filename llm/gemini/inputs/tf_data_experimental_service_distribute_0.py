
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_service_distribute_inputs():
    list_of_inputs = []

    # The persistent error "returns a function, but the input does not have inner values"
    # indicates the test harness needs to be told which tf.data.Dataset object to use.
    # The `distribute` function returns a transformation that is applied to a dataset.
    # After multiple failed attempts with keys like 'dataset', 'self', and 'x', this
    # attempt hypothesizes that the special key expected by the harness is 'input'.

    # Input 1: Basic 'distributed_epoch' mode
    input_dict_1 = {
        'input': np.arange(10, dtype=np.int64),
        'processing_mode': 'distributed_epoch',
        'service': 'grpc://localhost:8001',
        'job_name': 'job_1',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 10,
        'data_transfer_protocol': 'grpc',
        'compression': 'AUTO',
        'cross_trainer_cache': tuple(),
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 'parallel_epochs' mode with float data
    input_dict_2 = {
        'input': np.random.rand(5).astype(np.float32),
        'processing_mode': 'parallel_epochs',
        'service': 'grpc://localhost:8002',
        'job_name': 'job_2',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 5,
        'data_transfer_protocol': 'local',
        'compression': 'AUTO',
        'cross_trainer_cache': tuple(),
        'target_workers': 'LOCAL'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Coordinated read scenario with multiple consumers
    input_dict_3 = {
        'input': np.arange(20, dtype=np.int64),
        'processing_mode': 'parallel_epochs',
        'service': 'grpc://localhost:8003',
        'job_name': 'coordinated_job',
        'consumer_index': 1,
        'num_consumers': 2,
        'max_outstanding_requests': 8,
        'data_transfer_protocol': 'grpc',
        'compression': 'SNAPPY',
        'cross_trainer_cache': tuple(),
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 'DYNAMIC' sharding policy with string data
    input_dict_4 = {
        'input': np.array([b'record_a', b'record_b', b'record_c']),
        'processing_mode': 'DYNAMIC',
        'service': 'grpc://localhost:8004',
        'job_name': 'dynamic_job',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 16,
        'data_transfer_protocol': 'arrow',
        'compression': 'AUTO',
        'cross_trainer_cache': tuple(),
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 'OFF' sharding policy
    input_dict_5 = {
        'input': np.arange(8, dtype=np.int32),
        'processing_mode': 'OFF',
        'service': 'grpc://localhost:8005',
        'job_name': 'off_job',
        'consumer_index': 0,
        'num_consumers': 1,
        'max_outstanding_requests': 4,
        'data_transfer_protocol': 'grpc',
        'compression': 'SNAPPY',
        'cross_trainer_cache': tuple(),
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

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
