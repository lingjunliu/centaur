
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_experimental_async_clear_error_inputs():
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
generated_inputs["tf.experimental.async_clear_error"] = tf_experimental_async_clear_error_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.async_clear_error' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.async_clear_error'.")

check_valid('tf.experimental.async_clear_error', generated_inputs['tf.experimental.async_clear_error'], lib="tf", suffix=0)
