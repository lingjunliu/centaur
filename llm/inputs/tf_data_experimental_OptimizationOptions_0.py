
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_OptimizationOptions_inputs():
    list_of_inputs = []

    # Input 1: No options set
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: noop_elimination enabled
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: apply_default_optimizations disabled
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: autotune enabled
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: map_and_batch_fusion enabled
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: filter_fusion enabled
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: map_vectorization enabled
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: slack enabled
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: copy_to_device enabled
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: use_default_device enabled
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.OptimizationOptions"] = tf_data_experimental_OptimizationOptions_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.OptimizationOptions' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.OptimizationOptions'.")

check_valid('tf.data.experimental.OptimizationOptions', generated_inputs['tf.data.experimental.OptimizationOptions'], lib="tf", suffix=0)
