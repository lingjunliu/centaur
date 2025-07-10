
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_experimental_register_filesystem_plugin_inputs():
    list_of_inputs = []

    # Input 1: Empty string
    input_dict = {"plugin_location": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Non-existent file
    input_dict = {"plugin_location": "non_existent_file.so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Invalid path containing backslashes
    input_dict = {"plugin_location": "C:\\invalid\\path.so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Path with spaces
    input_dict = {"plugin_location": "path with spaces.so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  Path with unicode characters
    input_dict = {"plugin_location": "插件.so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Just a dot
    input_dict = {"plugin_location": "."}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Dot slash
    input_dict = {"plugin_location": "./"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Dot dot slash
    input_dict = {"plugin_location": "../"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Non existent absolute path
    input_dict = {"plugin_location": "/this/path/does/not/exist.so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  Non existent relative path in a deeper dir
    input_dict = {"plugin_location": "./deep/path/that/does/not/exist.so"}
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
