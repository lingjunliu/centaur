
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_noop_inputs():
    list_of_inputs = []

    name = "noop"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "noop_1"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "NoOpControl"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "_hidden_noop"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "test_noop_alpha"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "noopUpperCASE"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "n1234567890"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "noop__double__underscore"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "control_dep_noop_v2"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "op_x"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "placeholder_op_noop"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "noop_final_case"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.NoOp"] = tf_raw_ops_noop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.NoOp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NoOp'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.NoOp', generated_inputs['tf.raw_ops.NoOp'], lib="tf", suffix=0)
