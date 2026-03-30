
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_case_inputs():
    list_of_inputs = []

    # Input 1
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant(1)])]
    default = lambda: [tf.constant(0)]
    exclusive = False
    strict = False
    name = "case_1"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    pred_fn_pairs = [(tf.constant(False), lambda: [tf.constant(1)])]
    default = lambda: [tf.constant(0)]
    exclusive = False
    strict = False
    name = "case_2"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    pred_fn_pairs = [(tf.constant(False), lambda: [tf.constant(1)]), (tf.constant(True), lambda: [tf.constant(2)])]
    default = lambda: [tf.constant(0)]
    exclusive = False
    strict = False
    name = "case_3"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    pred_fn_pairs = [(tf.constant(False), lambda: [tf.constant(1)]), (tf.constant(False), lambda: [tf.constant(2)])]
    default = lambda: [tf.constant(0)]
    exclusive = False
    strict = False
    name = "case_4"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant(1)]), (tf.constant(True), lambda: [tf.constant(2)])]
    default = lambda: [tf.constant(0)]
    exclusive = True
    strict = False
    name = "case_5"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant([1, 2, 3])])]
    default = lambda: [tf.constant([0, 0, 0])]
    exclusive = False
    strict = False
    name = "case_6"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    pred_fn_pairs = [(tf.constant(False), lambda: [tf.constant([1, 2, 3])])]
    default = lambda: [tf.constant([0, 0, 0])]
    exclusive = False
    strict = False
    name = "case_7"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant(1), tf.constant(2)])]
    default = lambda: [tf.constant(0), tf.constant(0)]
    exclusive = False
    strict = False
    name = "case_8"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    pred_fn_pairs = [(tf.constant(False), lambda: [tf.constant(1), tf.constant(2)])]
    default = lambda: [tf.constant(0), tf.constant(0)]
    exclusive = False
    strict = False
    name = "case_9"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant(1.0)]), (tf.constant(False), lambda: [tf.constant(2.0)])]
    default = lambda: [tf.constant(0.0)]
    exclusive = False
    strict = False
    name = "case_10"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    pred_fn_pairs = [(tf.constant(True), lambda: [tf.constant([1,2,3], dtype=tf.int32)])]
    default = lambda: [tf.constant([0,0,0], dtype=tf.int32)]
    exclusive = False
    strict = True
    name = "case_11"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": [default], "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.case"] = tf_case_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.case' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.case'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.case', generated_inputs['tf.case'], lib="tf", suffix=0)
