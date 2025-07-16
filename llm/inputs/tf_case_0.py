
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_case_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default
    exclusive = False
    strict = False
    name = "case_1"
    input_dict = {"pred_fn_pairs": [(tf.constant(False), lambda: [tf.constant(1)])], "default": lambda: [tf.constant(3)], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: One predicate is True
    exclusive = False
    strict = False
    name = "case_2"
    input_dict = {"pred_fn_pairs": [(tf.constant(True), lambda: [tf.constant(10)])], "default": lambda: [tf.constant(30)], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Exclusive is True, only one True predicate
    exclusive = True
    strict = False
    name = "case_3"
    input_dict = {"pred_fn_pairs": [(tf.constant(True), lambda: [tf.constant(100)])], "default": lambda: [tf.constant(300)], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Nested tensors
    exclusive = False
    strict = False
    name = "case_4"
    input_dict = {"pred_fn_pairs": [(tf.constant(False), lambda: [tf.constant([[1, 2], [3, 4]])])], "default": lambda: [tf.constant([[9, 10], [11, 12]])], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple tensors returned
    exclusive = False
    strict = False
    name = "case_5"
    input_dict = {"pred_fn_pairs": [(tf.constant(True), lambda: [tf.constant(1), tf.constant(2)])], "default": lambda: [tf.constant(5), tf.constant(6)], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: No default
    exclusive = False
    strict = False
    name = "case_6"
    input_dict = {"pred_fn_pairs": [(tf.constant(True), lambda: [tf.constant(1)])], "default": None, "exclusive": exclusive, "strict": strict, "name": name}
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
