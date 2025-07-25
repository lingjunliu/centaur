
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

class _WorkerConfig:
    """
    A mock class to hold worker configuration. This is used to bypass
    potential issues in the testing environment with passing `namedtuple`
    objects, which might be causing the `AttributeError: 'tuple' object...`
    errors. The `tf.data.experimental.service.WorkerServer` should accept
    any object that has the required attributes (duck typing).
    """
    def __init__(self,
                 dispatcher_address,
                 port=0,
                 worker_address=None,
                 protocol="grpc",
                 data_transfer_protocol=None,
                 heartbeat_interval_ms=None,
                 dispatcher_timeout_ms=None):
        self.port = port
        self.dispatcher_address = dispatcher_address
        self.worker_address = worker_address
        self.protocol = protocol
        self.data_transfer_protocol = data_transfer_protocol
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.dispatcher_timeout_ms = dispatcher_timeout_ms

def tf_data_experimental_service_workerserver_inputs():
    list_of_inputs = []

    # Input 1: Basic config, start immediately
    config1 = _WorkerConfig(dispatcher_address="localhost:5050")
    input_dict_1 = {'config': config1, 'start': True}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic config, do not start immediately
    config2 = _WorkerConfig(dispatcher_address="localhost:5051")
    input_dict_2 = {'config': config2, 'start': False}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Specify a port for the worker
    config3 = _WorkerConfig(port=6000, dispatcher_address="127.0.0.1:8000")
    input_dict_3 = {'config': config3, 'start': True}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Let the system pick a port (port=0)
    config4 = _WorkerConfig(port=0, dispatcher_address="my-dispatcher-host:9090")
    input_dict_4 = {'config': config4, 'start': True}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: IPv6 address for dispatcher
    config5 = _WorkerConfig(dispatcher_address="[::1]:50051")
    input_dict_5 = {'config': config5, 'start': True}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Explicitly specify protocol
    config6 = _WorkerConfig(dispatcher_address="localhost:6060", protocol="grpc")
    input_dict_6 = {'config': config6, 'start': True}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Specify heartbeat interval
    config7 = _WorkerConfig(
        dispatcher_address="dns:///dispatcher-service.default.svc.cluster.local:5050",
        heartbeat_interval_ms=1000)
    input_dict_7 = {'config': config7, 'start': False}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Specify dispatcher timeout
    config8 = _WorkerConfig(
        dispatcher_address="192.168.1.100:7000",
        dispatcher_timeout_ms=5000)
    input_dict_8 = {'config': config8, 'start': True}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Specify data transfer protocol
    config9 = _WorkerConfig(
        dispatcher_address="tf-dispatcher:5050",
        data_transfer_protocol="local")
    input_dict_9 = {'config': config9, 'start': True}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Specify worker address explicitly
    config10 = _WorkerConfig(
        dispatcher_address="localhost:5050",
        worker_address="localhost:6001")
    input_dict_10 = {'config': config10, 'start': True}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.service.WorkerServer"] = tf_data_experimental_service_workerserver_inputs()

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
