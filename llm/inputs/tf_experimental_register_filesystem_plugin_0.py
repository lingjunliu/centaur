
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_experimental_register_filesystem_plugin_inputs():
    list_of_inputs = []

    # All of the following are removed because they do not represent existing files, which leads to FileNotFoundError. The API does not seem equipped to handle non-existent files gracefully.

    # Input 11: Just a filename
    #plugin_location = "some_plugin.so"
    #input_dict = {"plugin_location": plugin_location}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: More complex filename
    #plugin_location = "path/to/some_plugin.so"
    #input_dict = {"plugin_location": plugin_location}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Another filename
    #plugin_location = "./some_plugin.so"
    #input_dict = {"plugin_location": plugin_location}
    #list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 14: Unicode filename
    #plugin_location = "你好世界.so"
    #input_dict = {"plugin_location": plugin_location}
    #list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 15: filename with special characters
    #plugin_location = "plugin!@#$%.so"
    #input_dict = {"plugin_location": plugin_location}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Create dummy so file
    plugin_location = "dummy_plugin.so"
    with open(plugin_location, "w") as f:
      f.write("")
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.remove(plugin_location)
    
    plugin_location = "./dummy_plugin.so"
    with open("dummy_plugin.so", "w") as f:
      f.write("")
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))
    os.remove("dummy_plugin.so")

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
