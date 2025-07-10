
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_device_inputs():
    list_of_inputs = []

    # Input 1
    device_name = '/cpu:0'
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    device_name = '/gpu:0'
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    device_name = '/job:worker/task:1/device:cpu:0'
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    device_name = '/job:localhost/replica:0/task:0/device:GPU:0'
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    device_name = '/device:cpu:0'
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    device_name = '/device:GPU:1'
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    device_name = '/job:worker'
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    device_name = '/task:0'
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    device_name = '' # Empty string, which means no device specified
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    device_name = '/job:gpu_server/device:gpu:0'
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    device_name = '/replica:0'
    input_dict = {"device_name": device_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.device"] = tf_device_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.device'.")

check_valid('tf.device', generated_inputs['tf.device'], lib="tf", suffix=0)
