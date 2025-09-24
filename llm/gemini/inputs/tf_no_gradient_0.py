
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_no_gradient_inputs():
    list_of_inputs = []

    # Input 1
    op_type = "Size_1"
    input_dict = {"op_type": op_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    op_type = "Shape_1"
    input_dict = {"op_type": op_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    op_type = "Rank_1"
    input_dict = {"op_type": op_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    op_type = "Const_1"
    input_dict = {"op_type": op_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    op_type = "NoOp_1"
    input_dict = {"op_type": op_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    op_type = "Print_1"
    input_dict = {"op_type": op_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    op_type = "Placeholder_1"
    input_dict = {"op_type": op_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    op_type = "Identity_1"
    input_dict = {"op_type": op_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    op_type = "StopGradient_1"
    input_dict = {"op_type": op_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    op_type = "ResourceGather_1"
    input_dict = {"op_type": op_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.no_gradient"] = tf_no_gradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.no_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.no_gradient'.")

check_valid('tf.no_gradient', generated_inputs['tf.no_gradient'], lib="tf", suffix=0)
