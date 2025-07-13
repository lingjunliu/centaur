
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_experimental_register_filesystem_plugin_inputs():
    list_of_inputs = []

    # Input 1: Just a filename, assuming current directory is not in the system path.
    plugin_location = "my_plugin.so"
    input_dict = {"plugin_location": plugin_location}
    #list_of_inputs.append(copy.deepcopy(input_dict)) # Remove - triggers immediate error
    
    # Input 2: A very long path (likely invalid)
    plugin_location = "/tmp/" + "a" * 200 + ".so"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Path with multiple slashes
    plugin_location = "/tmp///my_plugin.so"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Path using tilde (~) for home directory. Requires expanding for valid OS path.
    plugin_location = "~/.my_plugin.so"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Path with a directory traversal (..). Assumes the previous directory doesn't contain a plugin file
    plugin_location = "../my_plugin.so"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.register_filesystem_plugin"] = tf_experimental_register_filesystem_plugin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.register_filesystem_plugin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.register_filesystem_plugin'.")

check_valid('tf.experimental.register_filesystem_plugin', generated_inputs['tf.experimental.register_filesystem_plugin'], lib="tf", suffix=0)
