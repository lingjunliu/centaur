
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_whole_file_reader_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "container": "",
        "shared_name": "",
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "container": "my_container",
        "shared_name": "my_shared_name",
        "name": "my_op_name"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "container": "another_container",
        "shared_name": "",
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "container": "",
        "shared_name": "another_shared_name",
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "container": "",
        "shared_name": "",
        "name": "another_op_name"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        "container": "container_123",
        "shared_name": "shared_123",
        "name": "name_123"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "container": "special_container",
        "shared_name": "",
        "name": "special_name"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        "container": "",
        "shared_name": "unique_shared",
        "name": "unique_name"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "container": "test_container",
        "shared_name": "test_shared",
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "container": "container4",
        "shared_name": "",
        "name": "name4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
temp_list = tf_raw_ops_whole_file_reader_inputs()
final_list = []
for input_dict in temp_list:
  final_input_dict = {}
  for k, v in input_dict.items():
    final_input_dict[k] = np.array(v, dtype=np.string_)
  final_list.append(final_input_dict)

generated_inputs["tf.raw_ops.WholeFileReader"] = final_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.WholeFileReader' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.WholeFileReader'.")

check_valid('tf.raw_ops.WholeFileReader', generated_inputs['tf.raw_ops.WholeFileReader'], lib="tf", suffix=0)
