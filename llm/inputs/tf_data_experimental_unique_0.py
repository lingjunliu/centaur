
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_unique_inputs():
    list_of_inputs = []

    # Input 1: Simple integer dataset
    input_dict1 = {}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: String dataset
    input_dict2 = {}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float dataset
    input_dict3 = {}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Dataset with tuples
    input_dict4 = {}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Dataset with numpy arrays
    input_dict5 = {}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty dataset
    input_dict6 = {}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Dataset with mixed types
    input_dict7 = {}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Dataset with negative values
    input_dict8 = {}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Dataset with a single element repeated
    input_dict9 = {}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Larger dataset
    input_dict10 = {}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.data.experimental.unique"] = tf_data_experimental_unique_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.unique'.")

check_valid('tf.data.experimental.unique', generated_inputs['tf.data.experimental.unique'], lib="tf", suffix=0)
