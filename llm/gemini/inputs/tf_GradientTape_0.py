
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_GradientTape_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "persistent": False,
        "watch_accessed_variables": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "persistent": True,
        "watch_accessed_variables": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "persistent": False,
        "watch_accessed_variables": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "persistent": True,
        "watch_accessed_variables": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "persistent": np.bool_(False),
        "watch_accessed_variables": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "persistent": np.bool_(True),
        "watch_accessed_variables": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "persistent": False,
        "watch_accessed_variables": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "persistent": True,
        "watch_accessed_variables": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "persistent": np.array(False),
        "watch_accessed_variables": np.array(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    input_dict = {
        "persistent": np.array(True),
        "watch_accessed_variables": np.array(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        "persistent": np.bool_(0),
        "watch_accessed_variables": np.bool_(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.GradientTape"] = tf_GradientTape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.GradientTape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.GradientTape'.")

check_valid('tf.GradientTape', generated_inputs['tf.GradientTape'], lib="tf", suffix=0)
