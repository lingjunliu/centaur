
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_data_experimental_service_WorkerServer_inputs():
    list_of_inputs = []

    # Input 1: Valid WorkerConfig
    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address="localhost:5000")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid WorkerConfig, start=False
    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address="127.0.0.1:5001")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid WorkerConfig, different dispatcher_address
    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address="example.com:6000")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Different ports
    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address="192.168.1.100:7000")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: different address
    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address="my-server:8000")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All zeros
    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address="0.0.0.0:9000")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: long address
    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address="some-long-address:10000")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different port 
    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address="another-address:11000")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Yet another address
    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address="yet-another-address:12000")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Final Address
    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address="final-address:13000")
    start = False
    input_dict = {"config": config, "start": start}
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
