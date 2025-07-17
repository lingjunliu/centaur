
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_case_inputs():
    list_of_inputs = []

    def f1():
        return [tf.convert_to_tensor(np.array(17), dtype=tf.int32)]

    def f2():
        return [tf.convert_to_tensor(np.array(23), dtype=tf.int32)]
    
    def f3():
        return [tf.convert_to_tensor(np.array(-1), dtype=tf.int32)]

    def f4():
        return [tf.convert_to_tensor(np.array(1.5), dtype=tf.float32)]
    
    def f5():
        return [tf.convert_to_tensor(np.array(2.5), dtype=tf.float32)]

    def f6():
        return [tf.convert_to_tensor(np.array(5), dtype=tf.int32)]

    def f7():
        return [tf.convert_to_tensor(np.array([1, 2]), dtype=tf.int32)]

    def f8():
        return [tf.convert_to_tensor(np.array([3, 4]), dtype=tf.int32)]


    # Input 1: Simple case with default
    pred_fn_pairs = [(tf.constant(False), f1)]
    default = f2
    exclusive = False
    strict = False
    name = "case_1"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple case without default
    pred_fn_pairs = [(tf.constant(True), f1)]
    default = f2
    exclusive = False
    strict = False
    name = "case_2"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple predicates, exclusive=True
    pred_fn_pairs = [(tf.constant(False), f1), (tf.constant(False), f2)]
    default = f3
    exclusive = True
    strict = False
    name = "case_3"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple predicates, exclusive=False
    pred_fn_pairs = [(tf.constant(True), f1), (tf.constant(True), f2)]
    default = f3
    exclusive = False
    strict = False
    name = "case_4"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data types
    pred_fn_pairs = [(tf.constant(True), f4)]
    default = f5
    exclusive = False
    strict = False
    name = "case_6"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: No default
    pred_fn_pairs = [(tf.constant(True), f1)]
    default = f2
    exclusive = False
    strict = False
    name = "case_7"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Empty pred_fn_pairs with a default
    pred_fn_pairs = []
    default = f6
    exclusive = False
    strict = False
    name = "case_8"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different shapes in output
    pred_fn_pairs = [(tf.constant(True), f7)]
    default = f8
    exclusive = False
    strict = False
    name = "case_9"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boolean False
    pred_fn_pairs = [(tf.constant(False), f1)]
    default = f2
    exclusive = False
    strict = False
    name = "case_10"
    input_dict = {"pred_fn_pairs": pred_fn_pairs, "default": default, "exclusive": exclusive, "strict": strict, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean False
    pred_fn_pairs = [(tf.constant(False), f1), (tf.constant(True), f2)]
    default = f3
    exclusive = False
    strict = False
    name = "case_11"
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
