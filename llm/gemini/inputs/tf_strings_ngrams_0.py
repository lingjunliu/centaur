
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_ngrams_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array(["A", "B", "C", "D"], dtype=np.object_)
    ngram_width = 2
    separator = " "
    pad_values = None
    padding_width = None
    preserve_short_sequences = False
    name = None

    input_dict = {
        "data": data,
        "ngram_width": ngram_width,
        "separator": separator,
        "pad_values": pad_values,
        "padding_width": padding_width,
        "preserve_short_sequences": preserve_short_sequences,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array(["TF", "and", "keras"], dtype=np.object_)
    ngram_width = 1
    separator = "_"
    pad_values = None
    padding_width = None
    preserve_short_sequences = False
    name = "my_ngrams"

    input_dict = {
        "data": data,
        "ngram_width": ngram_width,
        "separator": separator,
        "pad_values": pad_values,
        "padding_width": padding_width,
        "preserve_short_sequences": preserve_short_sequences,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array(["one", "two", "three"], dtype=np.object_)
    ngram_width = 2
    separator = "-"
    pad_values = ("START", "END")
    padding_width = 1
    preserve_short_sequences = False
    name = None

    input_dict = {
        "data": data,
        "ngram_width": ngram_width,
        "separator": separator,
        "pad_values": pad_values,
        "padding_width": padding_width,
        "preserve_short_sequences": preserve_short_sequences,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array(["a", "b"], dtype=np.object_)
    ngram_width = 3
    separator = "|"
    pad_values = "PAD"
    padding_width = 2
    preserve_short_sequences = True
    name = "test_ngrams"

    input_dict = {
        "data": data,
        "ngram_width": ngram_width,
        "separator": separator,
        "pad_values": pad_values,
        "padding_width": padding_width,
        "preserve_short_sequences": preserve_short_sequences,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    data = np.array([["hello", "world"], ["tensorflow", "rocks"]], dtype=np.object_)
    ngram_width = 2
    separator = "++"
    pad_values = None
    padding_width = None
    preserve_short_sequences = False
    name = None

    input_dict = {
        "data": data,
        "ngram_width": ngram_width,
        "separator": separator,
        "pad_values": pad_values,
        "padding_width": padding_width,
        "preserve_short_sequences": preserve_short_sequences,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([["x", "y", "z"], ["1", "2", "3"]], dtype=np.object_)
    ngram_width = 3
    separator = "***"
    pad_values = ("BEGIN", "FINISH")
    padding_width = 1
    preserve_short_sequences = True
    name = "multiline_ngram"

    input_dict = {
        "data": data,
        "ngram_width": ngram_width,
        "separator": separator,
        "pad_values": pad_values,
        "padding_width": padding_width,
        "preserve_short_sequences": preserve_short_sequences,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array(["short"], dtype=np.object_)
    ngram_width = 2
    separator = "_"
    pad_values = "P"
    padding_width = 1
    preserve_short_sequences = True
    name = None

    input_dict = {
        "data": data,
        "ngram_width": ngram_width,
        "separator": separator,
        "pad_values": pad_values,
        "padding_width": padding_width,
        "preserve_short_sequences": preserve_short_sequences,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array(["a", "b", "c", "d", "e"], dtype=np.object_)
    ngram_width = 4
    separator = " "
    pad_values = None
    padding_width = None
    preserve_short_sequences = True
    name = None

    input_dict = {
        "data": data,
        "ngram_width": ngram_width,
        "separator": separator,
        "pad_values": pad_values,
        "padding_width": padding_width,
        "preserve_short_sequences": preserve_short_sequences,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array(["single"], dtype=np.object_)
    ngram_width = 1
    separator = "-"
    pad_values = ("LEFT", "RIGHT")
    padding_width = 5
    preserve_short_sequences = True
    name = "padding_test"

    input_dict = {
        "data": data,
        "ngram_width": ngram_width,
        "separator": separator,
        "pad_values": pad_values,
        "padding_width": padding_width,
        "preserve_short_sequences": preserve_short_sequences,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([["A", "B"], ["C", "D"]], dtype=np.object_)
    ngram_width = 1
    separator = " "
    pad_values = None
    padding_width = None
    preserve_short_sequences = False
    name = None

    input_dict = {
        "data": data,
        "ngram_width": ngram_width,
        "separator": separator,
        "pad_values": pad_values,
        "padding_width": padding_width,
        "preserve_short_sequences": preserve_short_sequences,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.ngrams"] = tf_strings_ngrams_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.ngrams' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.ngrams'.")

check_valid('tf.strings.ngrams', generated_inputs['tf.strings.ngrams'], lib="tf", suffix=0)
