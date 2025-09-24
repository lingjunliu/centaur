
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import os
import sys
import tensorflow as tf

def tf_experimental_register_filesystem_plugin_inputs():
    """
    Generates a list of syntactically valid inputs for tf.experimental.register_filesystem_plugin.
    This function targets different types of OS-level errors that are subclasses of the
    documented OSError, as simply providing non-existent paths leads to a repetitive
    FileNotFoundError. In a standard test environment without actual filesystem plugins,
    an error is the expected outcome.
    """
    list_of_inputs = []

    # Paths that are directories, which should raise IsADirectoryError (subclass of OSError)
    # or a similar runtime error upon attempting to load.
    list_of_inputs.append({'plugin_location': '/tmp'})
    list_of_inputs.append({'plugin_location': '.'})
    
    # Paths to files that likely exist but are not shared libraries and/or may have
    # restricted read permissions, aiming for PermissionError or a RuntimeError.
    # The testing environment appears to be Linux-based from the tracebacks.
    if sys.platform.startswith('linux'):
        list_of_inputs.append({'plugin_location': '/etc/passwd'}) # Exists, readable, but not a plugin
        list_of_inputs.append({'plugin_location': '/proc/self/exe'}) # Exists, but is the python executable
        list_of_inputs.append({'plugin_location': '/root'}) # Likely to cause PermissionError
        list_of_inputs.append({'plugin_location': '/etc/shadow'}) # Likely to cause PermissionError

    # An empty string, which is an invalid path and should raise an error.
    list_of_inputs.append({'plugin_location': ''})

    # Plausible but non-existent paths, which will raise FileNotFoundError.
    list_of_inputs.append({'plugin_location': 'non_existent_plugin_path.so'})
    list_of_inputs.append({'plugin_location': '/no/such/dir/plugin.so'})

    # A path to /dev/null, an existing character device file, not a library.
    if os.path.exists('/dev/null'):
        list_of_inputs.append({'plugin_location': '/dev/null'})

    # Ensure we have at least 10 inputs by duplicating if necessary.
    i = 0
    while len(list_of_inputs) < 10:
        # Add more unique non-existent paths
        list_of_inputs.append({'plugin_location': f'/tmp/another_fake_plugin_{i}.so'})
        i += 1

    return list_of_inputs[:10]

generated_inputs["tf.experimental.register_filesystem_plugin"] = tf_experimental_register_filesystem_plugin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.register_filesystem_plugin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.register_filesystem_plugin'.")

check_valid('tf.experimental.register_filesystem_plugin', generated_inputs['tf.experimental.register_filesystem_plugin'], lib="tf", suffix=0)
