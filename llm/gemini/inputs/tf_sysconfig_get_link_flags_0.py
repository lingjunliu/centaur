
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_sysconfig_get_link_flags_inputs():
    list_of_inputs = []

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sysconfig.get_link_flags"] = tf_sysconfig_get_link_flags_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sysconfig.get_link_flags' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sysconfig.get_link_flags'.")

check_valid('tf.sysconfig.get_link_flags', generated_inputs['tf.sysconfig.get_link_flags'], lib="tf", suffix=0)
