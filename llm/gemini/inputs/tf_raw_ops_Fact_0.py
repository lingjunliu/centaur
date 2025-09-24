
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_raw_ops_Fact_inputs():
    list_of_inputs = []

    input_dict = {
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "Fact1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "AnotherFact"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "FactAboutFactorials"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "   Fact with spaces   "
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "123Fact"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "_Fact"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "Fact_"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "AveryLongFactNameThatMightStillWork"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Fact"] = tf_raw_ops_Fact_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Fact' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Fact'.")

check_valid('tf.raw_ops.Fact', generated_inputs['tf.raw_ops.Fact'], lib="tf", suffix=0)
