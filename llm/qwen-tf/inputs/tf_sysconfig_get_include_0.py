
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sysconfig_get_include_inputs():
    list_of_inputs = []
    for _ in range(12):
        input_dict = {}
        list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs["tf.sysconfig.get_include"] = tf_sysconfig_get_include_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sysconfig.get_include' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sysconfig.get_include'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sysconfig.get_include', generated_inputs['tf.sysconfig.get_include'], lib="tf", suffix=0)
