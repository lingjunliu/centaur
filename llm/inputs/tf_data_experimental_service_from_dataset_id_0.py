
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_service_from_dataset_id_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": ("grpc", "localhost:5000"),
        "dataset_id": 123,
        "element_spec": [tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None)],
        "job_name": "job1",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 10,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "processing_mode": "distributed_epoch",
        "service": ("grpc", "localhost:5001"),
        "dataset_id": 456,
        "element_spec": [tf.TensorSpec(shape=(10,), dtype=tf.float32, name=None)],
        "job_name": "job2",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 5,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "ANY"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": ("grpc", "localhost:5002"),
        "dataset_id": 789,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.string, name=None)],
        "job_name": "job3",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 20,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "LOCAL"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "processing_mode": "distributed_epoch",
        "service": ("grpc", "localhost:5003"),
        "dataset_id": 101,
        "element_spec": [tf.TensorSpec(shape=(None, None), dtype=tf.int32, name=None)],
        "job_name": "job4",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 1,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": ("grpc", "localhost:5004"),
        "dataset_id": 202,
        "element_spec": [tf.TensorSpec(shape=(3, 4, 5), dtype=tf.float64, name=None)],
        "job_name": "job5",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 8,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "ANY"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "processing_mode": "distributed_epoch",
        "service": ("grpc", "localhost:5005"),
        "dataset_id": 303,
        "element_spec": [tf.TensorSpec(shape=(2,), dtype=tf.bool, name=None)],
        "job_name": "job6",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 15,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "LOCAL"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": ("grpc", "localhost:5006"),
        "dataset_id": 404,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.complex64, name=None)],
        "job_name": "job7",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 3,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "processing_mode": "distributed_epoch",
        "service": ("grpc", "localhost:5007"),
        "dataset_id": 505,
        "element_spec": [tf.TensorSpec(shape=(5, 5), dtype=tf.uint8, name=None)],
        "job_name": "job8",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 12,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "ANY"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": ("grpc", "localhost:5008"),
        "dataset_id": 606,
        "element_spec": [tf.TensorSpec(shape=(1, 1, 1, 1), dtype=tf.int8, name=None)],
        "job_name": "job9",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 7,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "LOCAL"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "processing_mode": "distributed_epoch",
        "service": ("grpc", "localhost:5009"),
        "dataset_id": 707,
        "element_spec": [tf.TensorSpec(shape=(6,), dtype=tf.string, name=None)],
        "job_name": "job10",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 9,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11 - Different service format
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": ("http", "127.0.0.1:8080"),
        "dataset_id": 808,
        "element_spec": [tf.TensorSpec(shape=(2, 2), dtype=tf.int16, name=None)],
        "job_name": "job11",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 4,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "ANY"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 - Different sharding policy
    input_dict = {
        "processing_mode": "off", # or tf.data.experimental.service.ShardingPolicy.OFF
        "service": ("grpc", "localhost:5010"),
        "dataset_id": 909,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.float16, name=None)],
        "job_name": "job12",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 11,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "LOCAL"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13 - Empty job name
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": ("grpc", "localhost:5011"),
        "dataset_id": 1010,
        "element_spec": [tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None)],
        "job_name": "",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 6,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14 - consumer and num_consumers are both 0
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": ("grpc", "localhost:5012"),
        "dataset_id": 1111,
        "element_spec": [tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None)],
        "job_name": "job14",
        "consumer_index": None,
        "num_consumers": None,
        "max_outstanding_requests": 6,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "AUTO"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 15 - Only consumer index is provided
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": ("grpc", "localhost:5013"),
        "dataset_id": 1212,
        "element_spec": [tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None)],
        "job_name": "job15",
        "consumer_index": 1,
        "num_consumers": None,
        "max_outstanding_requests": 7,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "AUTO"
    }
    # list_of_inputs.append(copy.deepcopy(input_dict)) # This is supposed to cause an error
    
    # Input 16 - Only num_consumers is provided
    input_dict = {
        "processing_mode": "parallel_epochs",
        "service": ("grpc", "localhost:5014"),
        "dataset_id": 1313,
        "element_spec": [tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None)],
        "job_name": "job16",
        "consumer_index": None,
        "num_consumers": 2,
        "max_outstanding_requests": 8,
        "data_transfer_protocol": None,
        "cross_trainer_cache": [],
        "target_workers": "AUTO"
    }
    # list_of_inputs.append(copy.deepcopy(input_dict)) # This is supposed to cause an error

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
