
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_service_distribute_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'processing_mode': 'distributed_epoch',
        'service': 'grpc://localhost:5000',
        'job_name': 'job1',
        'consumer_index': None,
        'num_consumers': None,
        'max_outstanding_requests': None,
        'data_transfer_protocol': None,
        'compression': 'AUTO',
        'cross_trainer_cache': (),
        'target_workers': 'AUTO'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'processing_mode': 'parallel_epochs',
        'service': 'grpc://127.0.0.1:6000',
        'job_name': 'job2',
        'consumer_index': None,
        'num_consumers': None,
        'max_outstanding_requests': None,
        'data_transfer_protocol': None,
        'compression': None,
        'cross_trainer_cache': (),
        'target_workers': 'ANY'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.service.distribute"] = tf_data_experimental_service_distribute_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.service.distribute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.distribute'.")

check_valid('tf.data.experimental.service.distribute', generated_inputs['tf.data.experimental.service.distribute'], lib="tf", suffix=0)
