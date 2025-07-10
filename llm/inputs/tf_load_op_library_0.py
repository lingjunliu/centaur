
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_load_op_library_inputs():
    list_of_inputs = []

    # Input 1: Basic valid path (but will likely fail as the file doesn't exist, which is fine)
    input_dict = {"library_filename": "test_op.so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Relative path with directory (likely to fail)
    input_dict = {"library_filename": "./plugins/my_op.so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Absolute path (using a dummy file name)
    absolute_path = os.path.abspath("dummy_op.so")
    input_dict = {"library_filename": absolute_path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty string (should be handled gracefully)
    input_dict = {"library_filename": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Simple name (likely to fail)
    input_dict = {"library_filename": "simple_op.so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Path with "lib" prefix (common in some systems)
    input_dict = {"library_filename": "libmy_op.so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Path with a version number
    input_dict = {"library_filename": "my_op.1.2.3.so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using a .dll extension (Windows, but may still be a valid string)
    input_dict = {"library_filename": "my_op.dll"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Just the extension (should error gracefully)
    input_dict = {"library_filename": ".so"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A single character
    input_dict = {"library_filename": "a"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: A directory
    input_dict = {"library_filename": "."}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.load_op_library"] = tf_load_op_library_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.load_op_library' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.load_op_library'.")

check_valid('tf.load_op_library', generated_inputs['tf.load_op_library'], lib="tf", suffix=0)
