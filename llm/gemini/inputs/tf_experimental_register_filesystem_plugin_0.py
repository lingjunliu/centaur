
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os
import sys

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_register_filesystem_plugin_inputs():
    list_of_inputs = []

    # Create dummy plugin files for testing
    def create_dummy_plugin(path):
        if not os.path.exists(path):
            try:
                with open(path, "wb") as f:
                    f.write(os.urandom(1024 * 200))  # Write 200KB of random bytes, should be enough
            except OSError as e:
                print(f"Error creating dummy plugin {path}: {e}")
                return False
        return True

    dummy_plugin_path = "./test_plugin.so"
    if not create_dummy_plugin(dummy_plugin_path):
        return []

    dummy_plugin_path_abs = os.path.abspath("./another_test_plugin.so")
    if not create_dummy_plugin(dummy_plugin_path_abs):
        return []

    dummy_plugin_path_spaces = "./my plugin.so"
    if not create_dummy_plugin(dummy_plugin_path_spaces):
        return []
            
    dummy_plugin_path_chars = "./plugin-with_chars!.so"
    if not create_dummy_plugin(dummy_plugin_path_chars):
        return []
            
    dummy_plugin_path_levels = "./plugins/level1/level2/test_plugin.so"
    os.makedirs(os.path.dirname(dummy_plugin_path_levels), exist_ok=True)
    if not create_dummy_plugin(dummy_plugin_path_levels):
        return []
            
    dummy_plugin_path_levels_abs = os.path.abspath("./plugins/level1/level2/another_test_plugin.so")
    os.makedirs(os.path.dirname(dummy_plugin_path_levels_abs), exist_ok=True)
    if not create_dummy_plugin(dummy_plugin_path_levels_abs):
        return []

    home_plugin_path = os.path.expanduser("~/my_plugin.so") #expanded path
    if not create_dummy_plugin(home_plugin_path):
        return []

    long_plugin_path = "./a_very_long_plugin_name_with_underscores.so"
    if not create_dummy_plugin(long_plugin_path):
        return []

    parent_plugin_path = "../plugin.so"
    parent_dir = os.path.dirname(os.getcwd())
    parent_plugin_path_abs = os.path.join(parent_dir, "plugin.so")
    os.makedirs(os.path.dirname(parent_plugin_path_abs), exist_ok=True)
    if not create_dummy_plugin(parent_plugin_path_abs):
        return []

    dylib_plugin_path = "./my_plugin.dylib"
    if not create_dummy_plugin(dylib_plugin_path):
        return []

    # Input 1:  Basic valid path
    plugin_location = dummy_plugin_path
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Absolute path
    plugin_location = dummy_plugin_path_abs
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Path with spaces
    plugin_location = dummy_plugin_path_spaces
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Path with special characters
    plugin_location = dummy_plugin_path_chars
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Path with multiple directory levels (relative)
    plugin_location = dummy_plugin_path_levels
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Path with multiple directory levels (absolute)
    plugin_location = dummy_plugin_path_levels_abs
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Path using environment variables (if applicable in your setup, otherwise a regular path)
    plugin_location = home_plugin_path
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Plugin with a longer filename
    plugin_location = long_plugin_path
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Plugin location at the parent directory
    plugin_location = parent_plugin_path_abs
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Plugin with an extension other than .so (e.g., .dylib on macOS) - replace with a suitable extension for your system
    plugin_location = dylib_plugin_path
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.register_filesystem_plugin', generated_inputs['tf.experimental.register_filesystem_plugin'], lib="tf", suffix=0)
