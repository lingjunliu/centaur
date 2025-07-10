
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_no_op_inputs():
    list_of_inputs = []

    input_dict = {"name": "op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "placeholder_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "control_edge_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "dummy_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "no_action"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "do_nothing"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "idle"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "op_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "tf_no_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "some_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.no_op"] = tf_no_op_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.no_op' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.no_op'.")

check_valid('tf.no_op', generated_inputs['tf.no_op'], lib="tf", suffix=0)
