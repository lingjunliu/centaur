
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_load_op_library_inputs():
    list_of_inputs = []

    # Input 1: Empty string
    library_filename = ""
    input_dict = {"library_filename": library_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: A simple valid name (though likely non-existent)
    library_filename = "test_op.so"
    input_dict = {"library_filename": library_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A different extension
    library_filename = "another_op.dylib"
    input_dict = {"library_filename": library_filename}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Path with spaces (likely to fail)
    library_filename = "My Test Library.so"
    input_dict = {"library_filename": library_filename}
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
