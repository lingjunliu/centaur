
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_service_distribute_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": "localhost:5000",
        "job_name": "job1",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": None,
        "data_transfer_protocol": None,
        "compression": "AUTO",
        "cross_trainer_cache": (),
        "target_workers": "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "processing_mode": "distributed_epoch",
        "service": "grpc://localhost:5001",
        "job_name": "job2",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 10,
        "data_transfer_protocol": "grpc",
        "compression": "SNAPPY",
        "cross_trainer_cache": (),
        "target_workers": "ANY"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": "remote:localhost:5002",
        "job_name": "job3",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 100,
        "data_transfer_protocol": None,
        "compression": None,
        "cross_trainer_cache": (),
        "target_workers": "LOCAL"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "processing_mode": "distributed_epoch",
        "service": "localhost:5003",
        "job_name": "long_job_name",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 1,
        "data_transfer_protocol": None,
        "compression": "AUTO",
        "cross_trainer_cache": (),
        "target_workers": "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": "localhost:5004",
        "job_name": "job5",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 50,
        "data_transfer_protocol": "grpc",
        "compression": "SNAPPY",
        "cross_trainer_cache": (),
        "target_workers": "ANY"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "processing_mode": "distributed_epoch",
        "service": "localhost:5005",
        "job_name": "job6",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": None,
        "data_transfer_protocol": None,
        "compression": None,
        "cross_trainer_cache": (),
        "target_workers": "LOCAL"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": "localhost:5006",
        "job_name": "job7",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 20,
        "data_transfer_protocol": None,
        "compression": "AUTO",
        "cross_trainer_cache": (),
        "target_workers": "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "processing_mode": "distributed_epoch",
        "service": "localhost:5007",
        "job_name": "job8",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 1000,
        "data_transfer_protocol": None,
        "compression": "SNAPPY",
        "cross_trainer_cache": (),
        "target_workers": "ANY"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": "localhost:5008",
        "job_name": "job9",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 5,
        "data_transfer_protocol": None,
        "compression": None,
        "cross_trainer_cache": (),
        "target_workers": "LOCAL"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    input_dict = {
        "processing_mode": "distributed_epoch",
        "service": "localhost:5009",
        "job_name": "job10",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 25,
        "data_transfer_protocol": None,
        "compression": "AUTO",
        "cross_trainer_cache": (),
        "target_workers": "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.service.distribute"] = tf_data_service_distribute_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.service.distribute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.distribute'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.service.distribute', generated_inputs['tf.data.experimental.service.distribute'], lib="tf", suffix=0)
