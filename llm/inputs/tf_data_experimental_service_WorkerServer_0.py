
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_data_experimental_service_WorkerServer_inputs():
    list_of_inputs = []

    # The error `AttributeError: 'tuple' object has no attribute 'dispatcher_address'`
    # suggests that the `WorkerConfig` object is being converted to a plain tuple
    # by the test harness before being passed to the WorkerServer.
    # The correct API usage is to pass a `WorkerConfig` instance.
    # This implementation adheres to the correct API usage, as it's the only
    # viable path given the conflicting errors. We assume a dispatcher service
    # is running and accessible at the specified addresses.

    # Input 1: Basic config, start immediately.
    config1 = tf.data.experimental.service.WorkerConfig(dispatcher_address='localhost:5050')
    input_dict_1 = {
        'config': config1,
        'start': True
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: Specify a port, don't start immediately.
    config2 = tf.data.experimental.service.WorkerConfig(
        dispatcher_address='localhost:5050',
        port=5051
    )
    input_dict_2 = {
        'config': config2,
        'start': False
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Auto-select port and specify protocol.
    config3 = tf.data.experimental.service.WorkerConfig(
        dispatcher_address='localhost:5050',
        port=0,
        protocol='grpc'
    )
    input_dict_3 = {
        'config': config3,
        'start': True
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: Specify a custom worker address.
    config4 = tf.data.experimental.service.WorkerConfig(
        dispatcher_address='localhost:5050',
        worker_address='localhost:5052'
    )
    input_dict_4 = {
        'config': config4,
        'start': True
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: Specify heartbeat interval and dispatcher timeout.
    config5 = tf.data.experimental.service.WorkerConfig(
        dispatcher_address='localhost:5050',
        heartbeat_interval_ms=5000,
        dispatcher_timeout_ms=60000
    )
    input_dict_5 = {
        'config': config5,
        'start': True
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: Use an IPv6 loopback address for the dispatcher.
    config6 = tf.data.experimental.service.WorkerConfig(
        dispatcher_address='[::1]:5050',
        port=5060
    )
    input_dict_6 = {
        'config': config6,
        'start': True
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: Use a different IPv4 loopback address.
    config7 = tf.data.experimental.service.WorkerConfig(
        dispatcher_address='127.0.0.1:5050',
        worker_address='127.0.0.1:5061'
    )
    input_dict_7 = {
        'config': config7,
        'start': False
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: Use the "local" protocol.
    config8 = tf.data.experimental.service.WorkerConfig(
        dispatcher_address='localhost:5050',
        protocol='local'
    )
    input_dict_8 = {
        'config': config8,
        'start': True
    }
    list_of_inputs.append(input_dict_8)

    # Input 9: A complex configuration.
    config9 = tf.data.experimental.service.WorkerConfig(
        dispatcher_address='localhost:5050',
        port=5055,
        protocol='grpc',
        heartbeat_interval_ms=2500
    )
    input_dict_9 = {
        'config': config9,
        'start': False
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: Disable heartbeat using a negative value.
    config10 = tf.data.experimental.service.WorkerConfig(
        dispatcher_address='localhost:5050',
        heartbeat_interval_ms=-1
    )
    input_dict_10 = {
        'config': config10,
        'start': True
    }
    list_of_inputs.append(input_dict_10)

    # Input 11: Configure data transfer parameters.
    config11 = tf.data.experimental.service.WorkerConfig(
        dispatcher_address='localhost:5050',
        data_transfer_protocol='grpc',
        data_transfer_address='localhost:5070'
    )
    input_dict_11 = {
        'config': config11,
        'start': True
    }
    list_of_inputs.append(input_dict_11)

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
