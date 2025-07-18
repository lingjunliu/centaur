
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_service_WorkerServer_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.service.WorkerServer function.
    """
    list_of_inputs = []

    # The error "TypeError: WorkerConfig.__new__() got an unexpected keyword argument 'shutdown_quiet_period_ms'"
    # indicates that 'shutdown_quiet_period_ms' is not a valid argument for WorkerConfig.
    # The valid arguments are port, dispatcher_address, worker_address, protocol,
    # dispatcher_timeout_ms, heartbeat_interval_ms, data_transfer_protocol, and data_transfer_address.
    # We will remove the invalid argument and ensure all constructor calls are valid.
    # We will use port=0 to let the system choose a free port, and start=False to avoid hanging.

    # Input 1: Basic config
    config1 = tf.data.experimental.service.WorkerConfig(
        port=0,
        dispatcher_address='localhost:5050'
    )
    input_dict = {
        'config': config1,
        'start': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Config with a specific worker port
    config2 = tf.data.experimental.service.WorkerConfig(
        port=5001,
        dispatcher_address='localhost:5051'
    )
    input_dict = {
        'config': config2,
        'start': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Config with IPv4 loopback address
    config3 = tf.data.experimental.service.WorkerConfig(
        port=0,
        dispatcher_address='127.0.0.1:6000'
    )
    input_dict = {
        'config': config3,
        'start': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Config with IPv6 address
    config4 = tf.data.experimental.service.WorkerConfig(
        port=7001,
        dispatcher_address='[::1]:7000'
    )
    input_dict = {
        'config': config4,
        'start': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Config with a specific protocol
    config5 = tf.data.experimental.service.WorkerConfig(
        port=0,
        dispatcher_address='localhost:8000',
        protocol='grpc'
    )
    input_dict = {
        'config': config5,
        'start': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Config with a specific worker address
    config6 = tf.data.experimental.service.WorkerConfig(
        port=9090,
        dispatcher_address='tf-dispatcher.service.local:8080',
        worker_address='tf-worker-1.service.local:9090'
    )
    input_dict = {
        'config': config6,
        'start': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Config with a custom heartbeat interval
    config7 = tf.data.experimental.service.WorkerConfig(
        port=0,
        dispatcher_address='0.0.0.0:10000',
        heartbeat_interval_ms=5000
    )
    input_dict = {
        'config': config7,
        'start': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Config with a custom dispatcher timeout
    config8 = tf.data.experimental.service.WorkerConfig(
        port=0,
        dispatcher_address='localhost:49151',
        dispatcher_timeout_ms=10000
    )
    input_dict = {
        'config': config8,
        'start': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Config with data transfer protocol
    config9 = tf.data.experimental.service.WorkerConfig(
        port=0,
        dispatcher_address='dispatcher-server:12345',
        data_transfer_protocol='grpc'
    )
    input_dict = {
        'config': config9,
        'start': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Config with a combination of parameters
    config10 = tf.data.experimental.service.WorkerConfig(
        port=0,
        dispatcher_address='10.0.0.42:54321',
        protocol='grpc',
        heartbeat_interval_ms=2000,
        dispatcher_timeout_ms=5000
    )
    input_dict = {
        'config': config10,
        'start': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.service.WorkerServer"] = tf_data_experimental_service_WorkerServer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.service.WorkerServer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.WorkerServer'.")

check_valid('tf.data.experimental.service.WorkerServer', generated_inputs['tf.data.experimental.service.WorkerServer'], lib="tf", suffix=0)
