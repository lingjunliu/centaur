
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_prefetch_to_device_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "device": "/cpu:0",
        "buffer_size": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "device": "/cpu:0",
        "buffer_size": 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "device": "/cpu:0",
        "buffer_size": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "device": "/cpu:0",
        "buffer_size": 1000
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "device": "/cpu:0",
        "buffer_size": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "device": "/cpu:0",
        "buffer_size": 2048
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "device": "/cpu:0",
        "buffer_size": 64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "device": "/cpu:0",
        "buffer_size": 32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "device": "/cpu:0",
        "buffer_size": 128
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "device": "/cpu:0",
        "buffer_size": 256
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.prefetch_to_device"] = tf_data_experimental_prefetch_to_device_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.prefetch_to_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.prefetch_to_device'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.prefetch_to_device', generated_inputs['tf.data.experimental.prefetch_to_device'], lib="tf", suffix=0)
