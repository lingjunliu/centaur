
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_load_library_inputs():
    list_of_inputs = []

    # Input 1: Path to current working directory (exists, but no libtfkernel* files)
    library_location = "."
    input_dict = {"library_location": library_location}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.load_library"] = tf_load_library_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.load_library' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.load_library'.")

check_valid('tf.load_library', generated_inputs['tf.load_library'], lib="tf", suffix=0)
