
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_case_inputs():
    list_of_inputs = []

    # Input 1
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant(17)])]
    default = lambda: [tf.constant(23)]
    exclusive = False
    strict = False
    name = "case_1"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    pred_fn_pairs = [(tf.constant(False), lambda: [tf.constant(17)])]
    default = lambda: [tf.constant(23)]
    exclusive = False
    strict = False
    name = "case_2"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant(17)]), (tf.constant(False), lambda: [tf.constant(25)])]
    default = lambda: [tf.constant(23)]
    exclusive = False
    strict = False
    name = "case_3"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    pred_fn_pairs = [(tf.constant(False), lambda: [tf.constant(17)]), (tf.constant(True), lambda: [tf.constant(25)])]
    default = lambda: [tf.constant(23)]
    exclusive = False
    strict = False
    name = "case_4"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    pred_fn_pairs = [(tf.constant(False), lambda: [tf.constant(17)]), (tf.constant(False), lambda: [tf.constant(25)])]
    default = lambda: [tf.constant(23)]
    exclusive = False
    strict = False
    name = "case_5"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant(17, dtype=tf.float32)]), (tf.constant(False), lambda: [tf.constant(25, dtype=tf.float32)])]
    default = lambda: [tf.constant(23, dtype=tf.float32)]
    exclusive = False
    strict = False
    name = "case_6"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant([1, 2, 3])])]
    default = lambda: [tf.constant([4, 5, 6])]
    exclusive = False
    strict = False
    name = "case_7"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant(17), tf.constant(18)]), (tf.constant(False), lambda: [tf.constant(25), tf.constant(26)])]
    default = lambda: [tf.constant(23), tf.constant(24)]
    exclusive = False
    strict = False
    name = "case_8"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9

    pred_fn_pairs = [(tf.constant(False), lambda: [tf.constant(17)]), (tf.constant(False), lambda: [tf.constant(23)])]
    default = lambda: [tf.constant(-1)]
    exclusive = True
    strict = False
    name = "case_9"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant(17)]), (tf.constant(True), lambda: [tf.constant(25)])]
    default = lambda: [tf.constant(23)]
    exclusive = True
    strict = False
    name = "case_10"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.case"] = tf_case_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.case' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.case'.")

check_valid('tf.case', generated_inputs['tf.case'], lib="tf", suffix=0)
