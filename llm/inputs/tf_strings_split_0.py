
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_split_inputs():
    list_of_inputs = []

    # Input 1
    input_str = tf.constant("hello world")
    sep_str = tf.constant(" ")
    maxsplit_int = -1
    name_str = None

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_str = tf.constant(["hello world", "a b c"])
    sep_str = tf.constant(" ")
    maxsplit_int = -1
    name_str = "split_strings"

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_str = tf.constant("1<>2<><>3")
    sep_str = tf.constant("<>")
    maxsplit_int = -1
    name_str = None

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_str = tf.constant("a,b,c,d")
    sep_str = tf.constant(",")
    maxsplit_int = 2
    name_str = None

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_str = tf.constant("  hello   world  ")
    sep_str = tf.constant("")
    maxsplit_int = -1
    name_str = None

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_str = tf.constant("hello|world|again")
    sep_str = tf.constant("|")
    maxsplit_int = 1
    name_str = "limited_split"

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_str = tf.constant(["one two", "three four five"])
    sep_str = tf.constant(" ")
    maxsplit_int = 1
    name_str = None

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_str = tf.constant("sentence.with.multiple.dots")
    sep_str = tf.constant(".")
    maxsplit_int = -1
    name_str = None

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    input_str = tf.constant("a;b;c;d;e")
    sep_str = tf.constant(";")
    maxsplit_int = 3
    name_str = "limited_split_2"

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_str = tf.constant(["a,b", "c,d,e"])
    sep_str = tf.constant(",")
    maxsplit_int = -1
    name_str = None

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Different maxsplit
    input_str = tf.constant("apple banana orange")
    sep_str = tf.constant(" ")
    maxsplit_int = 2  # Splitting into at most 3 parts
    name_str = None

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty input string
    input_str = tf.constant("")
    sep_str = tf.constant("")
    maxsplit_int = -1
    name_str = None

    input_dict = {
        "input": input_str,
        "sep": sep_str,
        "maxsplit": maxsplit_int,
        "name": name_str
    }
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
