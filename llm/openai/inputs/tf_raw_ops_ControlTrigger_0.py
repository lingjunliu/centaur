
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_controltrigger_inputs():
    list_of_inputs = []

    input_dict = {"name": "control_trigger"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "ct_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "op_001"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "a"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "Zzz_Trigger_Name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "scope1/scope2/ct"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "under_score_leading__"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "_private_name_"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "with_digits_1234567890"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "NESTED_scope/inner/ControlTrigger_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "long_name_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "mixedCASE_Name_Op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ControlTrigger"] = tf_raw_ops_controltrigger_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ControlTrigger' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ControlTrigger'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ControlTrigger', generated_inputs['tf.raw_ops.ControlTrigger'], lib="tf", suffix=0)
