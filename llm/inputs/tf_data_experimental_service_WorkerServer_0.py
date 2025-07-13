
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_service_WorkerServer_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input using WorkerConfig
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="localhost:5000")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dispatcher address using WorkerConfig
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="127.0.0.1:5001")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Another dispatcher address using WorkerConfig
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="example.com:8080")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Config with different port using WorkerConfig
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="localhost:6000")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  Start as False, using WorkerConfig
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="localhost:5002")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: IP Address config, using WorkerConfig
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="192.168.1.100:7000")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: More complex config, using WorkerConfig
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="my-cluster.internal:9000")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  numeric config (should still be a string), using WorkerConfig
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="1234567890:1234")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  long config string, using WorkerConfig
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="a_very_long_config_string_that_should_still_work:5555")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different master address
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="different_master:5000")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Empty dispatcher address
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="")
    start = True
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
