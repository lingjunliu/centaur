
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_split_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = tf.constant("hello world")
    sep_tensor = tf.constant(" ")
    maxsplit = -1
    name = None
    input_dict = {"input": input_tensor, "sep": sep_tensor, "maxsplit": maxsplit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = tf.constant(["hello world", "a b c"])
    sep_tensor = tf.constant(" ")
    maxsplit = -1
    name = "split_op"
    input_dict = {"input": input_tensor, "sep": sep_tensor, "maxsplit": maxsplit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = tf.constant("1<>2<><>3")
    sep_tensor = tf.constant("<>")
    maxsplit = -1
    name = None
    input_dict = {"input": input_tensor, "sep": sep_tensor, "maxsplit": maxsplit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = tf.constant("a,b,c,d")
    sep_tensor = tf.constant(",")
    maxsplit = 2
    name = None
    input_dict = {"input": input_tensor, "sep": sep_tensor, "maxsplit": maxsplit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = tf.constant(["one two", "three four five", "six"])
    sep_tensor = tf.constant(" ")
    maxsplit = 1
    name = None
    input_dict = {"input": input_tensor, "sep": sep_tensor, "maxsplit": maxsplit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = tf.constant("  leading and trailing spaces  ")
    sep_tensor = tf.constant("", dtype=tf.string)
    maxsplit = -1
    name = None
    input_dict = {"input": input_tensor, "sep": sep_tensor, "maxsplit": maxsplit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = tf.constant(["  leading spaces", "trailing spaces  "])
    sep_tensor = tf.constant("", dtype=tf.string)
    maxsplit = -1
    name = None
    input_dict = {"input": input_tensor, "sep": sep_tensor, "maxsplit": maxsplit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = tf.constant("multiple,,consecutive,,separators")
    sep_tensor = tf.constant(",")
    maxsplit = -1
    name = None
    input_dict = {"input": input_tensor, "sep": sep_tensor, "maxsplit": maxsplit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = tf.constant("no separator")
    sep_tensor = tf.constant(",")
    maxsplit = -1
    name = None
    input_dict = {"input": input_tensor, "sep": sep_tensor, "maxsplit": maxsplit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = tf.constant(["", ""])
    sep_tensor = tf.constant(" ", dtype=tf.string)
    maxsplit = -1
    name = None
    input_dict = {"input": input_tensor, "sep": sep_tensor, "maxsplit": maxsplit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.split"] = tf_strings_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.split'.")

check_valid('tf.strings.split', generated_inputs['tf.strings.split'], lib="tf", suffix=0)
