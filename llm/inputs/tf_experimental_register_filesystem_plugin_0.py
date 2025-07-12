
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_experimental_register_filesystem_plugin_inputs():
    list_of_inputs = []

    # Create a dummy file for testing.  The file needs to be a valid shared object,
    # but for testing purposes, we'll create an empty file.  This will likely result
    # in a different error later (e.g., during loading), but it satisfies the
    # immediate requirement that the file exists.
    dummy_plugin_name = "dummy_plugin.so"
    if not os.path.exists(dummy_plugin_name):
        # Create a dummy C file
        c_code = """
        #include <stdio.h>

        void dummy_function() {
            printf("Dummy function called\\n");
        }
        """
        with open("dummy.c", "w") as f:
            f.write(c_code)

        # Compile the C file into a shared object (Linux)
        compile_command = "gcc -shared -o dummy_plugin.so -fPIC dummy.c"
        result = os.system(compile_command)

        if result != 0:
            print(f"Compilation failed with code {result}")
            return []  # Stop if compilation fails

        if not os.path.exists(dummy_plugin_name): # Check if file creation was successful
            print(f"Plugin file {dummy_plugin_name} was not created.")
            return []  # if not, return an empty list since further tests would also fail.

    # Input 1: Valid path (assuming a dummy plugin exists at this relative location)
    plugin_location = dummy_plugin_name
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Absolute path (assuming a dummy plugin exists at this absolute location)
    abs_path = os.path.abspath(dummy_plugin_name)
    plugin_location = abs_path
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different valid relative path (assuming it exists)
    plugin_location = "plugins/dummy_custom_fs.so"
    if not os.path.exists("plugins/dummy_custom_fs.so"):
        os.makedirs("plugins", exist_ok=True)
        # Create dummy C file within plugins dir if doesn't exist
        c_code = """
        #include <stdio.h>

        void dummy_function() {
            printf("Dummy function called\\n");
        }
        """
        with open("plugins/dummy.c", "w") as f:
            f.write(c_code)
        compile_command = "gcc -shared -o plugins/dummy_custom_fs.so -fPIC plugins/dummy.c"
        result = os.system(compile_command)

        if result != 0:
            print(f"Compilation failed with code {result}")
            return []

        if not os.path.exists("plugins/dummy_custom_fs.so"):
            print(f"Plugin file plugins/dummy_custom_fs.so was not created.")
            return []  # if not, return an empty list since further tests would also fail.

    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Very long path (assuming it exists)
    long_path = "a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z/plugins/dummy_very_long_plugin.so"
    if not os.path.exists(long_path):
        os.makedirs(os.path.dirname(long_path), exist_ok=True)
        c_code = """
        #include <stdio.h>

        void dummy_function() {
            printf("Dummy function called\\n");
        }
        """
        with open("long_dummy.c", "w") as f:
            f.write(c_code)
        compile_command = f"gcc -shared -o {long_path} -fPIC long_dummy.c"
        result = os.system(compile_command)
        if result != 0:
            print(f"Compilation failed with code {result}")
            return []
        if not os.path.exists(long_path):
            print(f"Plugin file {long_path} was not created.")
            return []  # if not, return an empty list since further tests would also fail.

    plugin_location = long_path
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Path with spaces (assuming it exists)
    plugin_location = "my plugins/dummy custom file system.so"
    if not os.path.exists(plugin_location):
        os.makedirs("my plugins", exist_ok=True)
        # Create a dummy c file
        c_code = """
        #include <stdio.h>

        void dummy_function() {
            printf("Dummy function called\\n");
        }
        """
        with open("space_dummy.c", "w") as f:
            f.write(c_code)
        compile_command = f"gcc -shared -o \"{plugin_location}\" -fPIC space_dummy.c"
        result = os.system(compile_command)
        if result != 0:
            print(f"Compilation failed with code {result}")
            return []
        if not os.path.exists(plugin_location):
            print(f"Plugin file {plugin_location} was not created.")
            return []  # if not, return an empty list since further tests would also fail.
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Path with special characters (assuming it exists)
    plugin_location = "plugins_with_$peci@l_chars_dummy.so"
    if not os.path.exists(plugin_location):
        c_code = """
        #include <stdio.h>

        void dummy_function() {
            printf("Dummy function called\\n");
        }
        """
        with open("special_dummy.c", "w") as f:
            f.write(c_code)
        compile_command = f"gcc -shared -o {plugin_location} -fPIC special_dummy.c"
        result = os.system(compile_command)
        if result != 0:
            print(f"Compilation failed with code {result}")
            return []
        if not os.path.exists(plugin_location):
            print(f"Plugin file {plugin_location} was not created.")
            return []  # if not, return an empty list since further tests would also fail.

    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Path with uppercase characters (assuming it exists)
    plugin_location = "DummyMyPlugin.SO"
    if not os.path.exists(plugin_location):
        c_code = """
        #include <stdio.h>

        void dummy_function() {
            printf("Dummy function called\\n");
        }
        """
        with open("upper_dummy.c", "w") as f:
            f.write(c_code)
        compile_command = f"gcc -shared -o {plugin_location} -fPIC upper_dummy.c"
        result = os.system(compile_command)

        if result != 0:
            print(f"Compilation failed with code {result}")
            return []
        if not os.path.exists(plugin_location):
            print(f"Plugin file {plugin_location} was not created.")
            return []  # if not, return an empty list since further tests would also fail.
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Path with digits in the name (assuming it exists)
    plugin_location = "plugin123_dummy.so"
    if not os.path.exists(plugin_location):
        c_code = """
        #include <stdio.h>

        void dummy_function() {
            printf("Dummy function called\\n");
        }
        """
        with open("digit_dummy.c", "w") as f:
            f.write(c_code)
        compile_command = f"gcc -shared -o {plugin_location} -fPIC digit_dummy.c"
        result = os.system(compile_command)
        if result != 0:
            print(f"Compilation failed with code {result}")
            return []
        if not os.path.exists(plugin_location):
            print(f"Plugin file {plugin_location} was not created.")
            return []  # if not, return an empty list since further tests would also fail.

    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Path with different extension (assuming it exists and works as a plugin)
    plugin_location = "dummy_plugin.so"
    input_dict = {"plugin_location": plugin_location}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Another valid relative path (assuming it exists). Using parent directory.
    plugin_location = "../dummy_another_plugin.so"
    parent_dir = os.path.dirname(os.getcwd())

    if not os.path.exists(os.path.join(parent_dir, "dummy_another_plugin.so")):
        c_code = """
        #include <stdio.h>

        void dummy_function() {
            printf("Dummy function called\\n");
        }
        """
        with open("parent_dummy.c", "w") as f:
            f.write(c_code)
        compile_command = f"gcc -shared -o {os.path.join(parent_dir, 'dummy_another_plugin.so')} -fPIC parent_dummy.c"
        result = os.system(compile_command)

        if result != 0:
            print(f"Compilation failed with code {result}")
            return []

        if not os.path.exists(os.path.join(parent_dir, "dummy_another_plugin.so")):
            print(f"Plugin file {os.path.join(parent_dir, 'dummy_another_plugin.so')} was not created.")
            return []  # if not, return an empty list since further tests would also fail.

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
