
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_service_WorkerServer_inputs():
    list_of_inputs = []

    # Input 1, valid
    worker_config = tf.data.experimental.service.WorkerConfig(dispatcher_address="localhost:5000")
    config = worker_config
    start = np.bool_(True).item()
    input_dict = {
        "config": config,
        "start": start
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    worker_config = tf.data.experimental.service.WorkerConfig(dispatcher_address="127.0.0.1:5001")
    config = worker_config
    start = np.bool_(False).item()
    input_dict = {
        "config": config,
        "start": start
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    worker_config = tf.data.experimental.service.WorkerConfig(dispatcher_address="example.com:8080")
    config = worker_config
    start = np.bool_(True).item()
    input_dict = {
        "config": config,
        "start": start
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    worker_config = tf.data.experimental.service.WorkerConfig(dispatcher_address="my-service:9000")
    config = worker_config
    start = np.bool_(False).item()
    input_dict = {
        "config": config,
        "start": start
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    worker_config = tf.data.experimental.service.WorkerConfig(dispatcher_address="some-other-service:7777")
    config = worker_config
    start = np.bool_(True).item()
    input_dict = {
        "config": config,
        "start": start
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    worker_config = tf.data.experimental.service.WorkerConfig(dispatcher_address="yet-another-service:1234")
    config = worker_config
    start = np.bool_(False).item()
    input_dict = {
        "config": config,
        "start": start
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    worker_config = tf.data.experimental.service.WorkerConfig(dispatcher_address="long-service-name:6666")
    config = worker_config
    start = np.bool_(True).item()
    input_dict = {
        "config": config,
        "start": start
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    worker_config = tf.data.experimental.service.WorkerConfig(dispatcher_address="short:5555")
    config = worker_config
    start = np.bool_(False).item()
    input_dict = {
        "config": config,
        "start": start
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    worker_config = tf.data.experimental.service.WorkerConfig(dispatcher_address="test-service:4444")
    config = worker_config
    start = np.bool_(True).item()
    input_dict = {
        "config": config,
        "start": start
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    worker_config = tf.data.experimental.service.WorkerConfig(dispatcher_address="final-service:3333")
    config = worker_config
    start = np.bool_(False).item()
    input_dict = {
        "config": config,
        "start": start
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.service.WorkerServer"] = tf_data_experimental_service_WorkerServer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.service.WorkerServer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.WorkerServer'.")

check_valid('tf.data.experimental.service.WorkerServer', generated_inputs['tf.data.experimental.service.WorkerServer'], lib="tf", suffix=0)
