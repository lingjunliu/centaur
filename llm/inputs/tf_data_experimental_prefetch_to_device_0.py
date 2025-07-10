
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_prefetch_to_device_inputs():
    list_of_inputs = []

    # Input 1: Basic CPU prefetch
    device = "/cpu:0"
    buffer_size = 10
    input_dict = {"device": device, "buffer_size": buffer_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic GPU prefetch (assuming GPU is available)
    try:
        device = "/gpu:0"
    except tf.errors.NotFoundError:
        device = "/cpu:0"
    buffer_size = 20
    input_dict = {"device": device, "buffer_size": buffer_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small buffer size
    device = "/cpu:0"
    buffer_size = 1
    input_dict = {"device": device, "buffer_size": buffer_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large buffer size
    device = "/cpu:0"
    buffer_size = 100
    input_dict = {"device": device, "buffer_size": buffer_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different CPU device name
    device = "/job:localhost/replica:0/task:0/device:CPU:0"
    buffer_size = 5
    input_dict = {"device": device, "buffer_size": buffer_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Device name with underscores
    device = "/device:CPU:0"
    buffer_size = 15
    input_dict = {"device": device, "buffer_size": buffer_size}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another GPU device name (if GPUs exist)
    try:
        device = "/gpu:1"
    except tf.errors.NotFoundError:
        device = "/cpu:0"

    buffer_size = 30
    input_dict = {"device": device, "buffer_size": buffer_size}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.prefetch_to_device"] = tf_data_experimental_prefetch_to_device_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.prefetch_to_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.prefetch_to_device'.")

check_valid('tf.data.experimental.prefetch_to_device', generated_inputs['tf.data.experimental.prefetch_to_device'], lib="tf", suffix=0)
