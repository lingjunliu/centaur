
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_data_experimental_copy_to_device_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "target_device": "/device:GPU:0",
        "source_device": "/device:CPU:0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "target_device": "/device:GPU:1",
        "source_device": "/device:CPU:0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "target_device": "/device:TPU:0",
        "source_device": "/device:CPU:0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "target_device": "/device:GPU:0",
        "source_device": "/device:GPU:1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "target_device": "/device:CPU:1",
        "source_device": "/device:CPU:0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "target_device": "/device:GPU:2",
        "source_device": "/device:GPU:0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        "target_device": "/device:CPU:0",
        "source_device": "/device:GPU:0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "target_device": "/device:GPU:0",
        "source_device": "/device:TPU:0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "target_device": "/device:CPU:1",
        "source_device": "/device:TPU:0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "target_device": "/device:GPU:3",
        "source_device": "/device:GPU:2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        "target_device": "/job:localhost/replica:0/task:0/device:GPU:0",
        "source_device": "/job:localhost/replica:0/task:0/device:CPU:0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.copy_to_device"] = tf_data_experimental_copy_to_device_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.copy_to_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.copy_to_device'.")

check_valid('tf.data.experimental.copy_to_device', generated_inputs['tf.data.experimental.copy_to_device'], lib="tf", suffix=0)
