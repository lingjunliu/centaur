
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_WholeFileReader_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.WholeFileReader operation.
    The 'RuntimeError: whole_file_reader op does not support eager execution' is an
    inherent limitation of this specific TensorFlow operation. It is designed to work
    only within a TensorFlow Graph (like in TF1.x or a tf.function), not in the default
    eager execution mode. The inputs provided here are syntactically correct for the
    operation's intended graph-based usage.
    """
    list_of_inputs = []

    # Input 1: Default values for container and shared_name
    input_dict = {
        'container': '',
        'shared_name': '',
        'name': 'reader1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Only container specified
    input_dict = {
        'container': 'my_container',
        'shared_name': '',
        'name': 'reader2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Only shared_name specified
    input_dict = {
        'container': '',
        'shared_name': 'my_shared_reader',
        'name': 'reader3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both container and shared_name specified
    input_dict = {
        'container': 'another_container',
        'shared_name': 'another_shared_reader',
        'name': 'reader4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All parameters are empty strings
    input_dict = {
        'container': '',
        'shared_name': '',
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Names containing numbers
    input_dict = {
        'container': 'container_v1',
        'shared_name': 'reader_2024',
        'name': 'op_99'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed-case names
    input_dict = {
        'container': 'ProductionContainer',
        'shared_name': 'GlobalFileReader',
        'name': 'ReadOperation'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Long, descriptive names
    input_dict = {
        'container': 'long_container_name_for_resource_isolation',
        'shared_name': 'shared_reader_instance_for_all_pipelines',
        'name': 'whole_file_reader_initialization_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Using a shared name in the same container with a different op name
    input_dict = {
        'container': 'another_container',
        'shared_name': 'another_shared_reader',
        'name': 'reader5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single character names
    input_dict = {
        'container': 'a',
        'shared_name': 'b',
        'name': 'c'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.WholeFileReader"] = tf_raw_ops_WholeFileReader_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.WholeFileReader' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.WholeFileReader'.")

check_valid('tf.raw_ops.WholeFileReader', generated_inputs['tf.raw_ops.WholeFileReader'], lib="tf", suffix=0)
