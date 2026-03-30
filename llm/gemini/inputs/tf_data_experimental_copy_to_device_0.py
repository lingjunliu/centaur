
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_copy_to_device_inputs():
    list_of_inputs = []

    # Input 1
    target_device = '/GPU:0'
    source_device = '/CPU:0'
    input_dict = {
        "target_device": target_device,
        "source_device": source_device
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    target_device = '/TPU:0'
    source_device = '/CPU:0'
    input_dict = {
        "target_device": target_device,
        "source_device": source_device
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    target_device = '/device:CPU:0'
    source_device = '/device:CPU:0'
    input_dict = {
        "target_device": target_device,
        "source_device": source_device
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    target_device = '/device:GPU:0'
    source_device = '/device:CPU:0'
    input_dict = {
        "target_device": target_device,
        "source_device": source_device
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    target_device = '/job:worker/replica:0/task:0/device:GPU:0'
    source_device = '/device:CPU:0'
    input_dict = {
        "target_device": target_device,
        "source_device": source_device
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    target_device = '/job:localhost/replica:0/task:0/device:CPU:0'
    source_device = '/device:GPU:0'
    input_dict = {
        "target_device": target_device,
        "source_device": source_device
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    target_device = '/physical_device:GPU:0'
    source_device = '/device:CPU:0'
    input_dict = {
        "target_device": target_device,
        "source_device": source_device
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    target_device = '/physical_device:CPU:0'
    source_device = '/device:GPU:0'
    input_dict = {
        "target_device": target_device,
        "source_device": source_device
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    target_device = '/device:XLA_GPU:0'
    source_device = '/device:XLA_CPU:0'
    input_dict = {
        "target_device": target_device,
        "source_device": source_device
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    target_device = '/device:XLA_CPU:0'
    source_device = '/device:CPU:0'
    input_dict = {
        "target_device": target_device,
        "source_device": source_device
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.copy_to_device"] = tf_data_experimental_copy_to_device_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.copy_to_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.copy_to_device'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.copy_to_device', generated_inputs['tf.data.experimental.copy_to_device'], lib="tf", suffix=0)
