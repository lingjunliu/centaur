
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_service_WorkerServer_inputs():
    list_of_inputs = []

    # Input 1
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="localhost:5000")
    start = np.array(True, dtype=np.bool_)
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="127.0.0.1:5001")
    start = np.array(False, dtype=np.bool_)
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="example.com:8080")
    start = np.array(True, dtype=np.bool_)
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="mydispatcher:1234")
    start = np.array(False, dtype=np.bool_)
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="0.0.0.0:8888")
    start = np.array(True, dtype=np.bool_)
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="long_dispatcher_name_0123456789:9999")
    start = np.array(False, dtype=np.bool_)
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="some_remote_server:5005")
    start = np.array(True, dtype=np.bool_)
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="another_server:5006")
    start = np.array(False, dtype=np.bool_)
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="a_third_server:5007")
    start = np.array(True, dtype=np.bool_)
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    config = tf.data.experimental.service.WorkerConfig(dispatcher_address="yet_another_server:5008")
    start = np.array(False, dtype=np.bool_)
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
