
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_scan_inputs():
    list_of_inputs = []

    # Input 1
    initial_state = np.array(0, dtype=np.int32)
    def scan_func1(state, element):
      return (state + element, element)
    input_dict = {"initial_state": initial_state, "scan_func": [scan_func1]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.scan"] = tf_data_experimental_scan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.scan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.scan'.")

check_valid('tf.data.experimental.scan', generated_inputs['tf.data.experimental.scan'], lib="tf", suffix=0)
