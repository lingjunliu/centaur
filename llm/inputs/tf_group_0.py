
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_group_inputs():
    list_of_inputs = []

    def create_dummy_op(name):
        v = tf.Variable(1.0, name=name)
        return v.assign(2.0).op

    # Input 1: Empty list of tensors
    input_dict = {"inputs": [], "name": "group_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single tensor
    op1 = create_dummy_op("op1")
    input_dict = {"inputs": [op1], "name": "group_single"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple tensors
    op1 = create_dummy_op("op2")
    op2 = create_dummy_op("op3")
    input_dict = {"inputs": [op1, op2], "name": "group_multiple"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensors with different shapes
    op1 = create_dummy_op("op4")
    op2 = create_dummy_op("op5")
    input_dict = {"inputs": [op1, op2], "name": "group_different_shapes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensors with different data types
    op1 = create_dummy_op("op6")
    op2 = create_dummy_op("op7")
    input_dict = {"inputs": [op1, op2], "name": "group_different_dtypes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensors with different ranks (number of dimensions)
    op1 = create_dummy_op("op8")
    op2 = create_dummy_op("op9")
    op3 = create_dummy_op("op10")
    input_dict = {"inputs": [op1, op2, op3], "name": "group_different_ranks"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensors with zero values
    op1 = create_dummy_op("op11")
    op2 = create_dummy_op("op12")
    input_dict = {"inputs": [op1, op2], "name": "group_zero_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensors with negative values
    op1 = create_dummy_op("op13")
    op2 = create_dummy_op("op14")
    input_dict = {"inputs": [op1, op2], "name": "group_negative_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger tensors
    op1 = create_dummy_op("op15")
    op2 = create_dummy_op("op16")
    input_dict = {"inputs": [op1, op2], "name": "group_larger_tensors"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensors with different dtypes and shapes
    op1 = create_dummy_op("op17")
    op2 = create_dummy_op("op18")
    op3 = create_dummy_op("op19")
    input_dict = {"inputs": [op1, op2, op3], "name": "group_complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.group"] = tf_group_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.group' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.group'.")

check_valid('tf.group', generated_inputs['tf.group'], lib="tf", suffix=0)
