
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_data_experimental_service_WorkerServer_inputs():
    list_of_inputs = []

    # Input 1: Using WorkerConfig
    config = tf.compat.v1.data.experimental.service.WorkerConfig(
        dispatcher_address="localhost:5000")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Using WorkerConfig
    config = tf.compat.v1.data.experimental.service.WorkerConfig(
        dispatcher_address="127.0.0.1:5001")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using WorkerConfig with different port
    config = tf.compat.v1.data.experimental.service.WorkerConfig(
        dispatcher_address="0.0.0.0:5002")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Using WorkerConfig with a different address
    config = tf.compat.v1.data.experimental.service.WorkerConfig(
        dispatcher_address="example.com:5003")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using WorkerConfig with a different address
    config = tf.compat.v1.data.experimental.service.WorkerConfig(
        dispatcher_address="myserver:5004")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Using WorkerConfig
    config = tf.compat.v1.data.experimental.service.WorkerConfig(
        dispatcher_address="anotherserver:5005")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Using WorkerConfig
    config = tf.compat.v1.data.experimental.service.WorkerConfig(
        dispatcher_address="yetanotherserver:5006")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using WorkerConfig
    config = tf.compat.v1.data.experimental.service.WorkerConfig(
        dispatcher_address="lastserver:5007")
    start = False
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Using WorkerConfig
    config = tf.compat.v1.data.experimental.service.WorkerConfig(
        dispatcher_address="testserver:5008")
    start = True
    input_dict = {"config": config, "start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Using WorkerConfig
    config = tf.compat.v1.data.experimental.service.WorkerConfig(
        dispatcher_address="finalserver:5009")
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
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.service.WorkerServer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.service.WorkerServer'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.service.WorkerServer', generated_inputs['tf.data.experimental.service.WorkerServer'], lib="tf", suffix=0)
