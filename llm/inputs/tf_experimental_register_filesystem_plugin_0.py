
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_experimental_register_filesystem_plugin_inputs():
    list_of_inputs = []

    # Input 1:  Invalid path (just a filename with .txt extension)
    plugin_location = "myplugin.txt"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Invalid path (just a filename with .dll extension)
    plugin_location = "myplugin.dll"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Invalid path (a directory)
    plugin_location = "mydir/"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Invalid path (an absolute directory)
    plugin_location = "/tmp/"  # Assuming /tmp exists
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Invalid path (just a filename without extension)
    plugin_location = "myplugin"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Invalid path (path with multiple slashes and no filename)
    plugin_location = "path/to/nowhere/"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Invalid path (relative path with "..", not pointing to a valid .so)
    plugin_location = "../myplugin.so"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Invalid path (filename with spaces and no extension)
    plugin_location = "my plugin"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Invalid path (filename starting with a dot and with invalid extension)
    plugin_location = ".myplugin.txt"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty string
    plugin_location = ""
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
