
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_StringNGrams_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array(["hello", "world", "tensorflow"], dtype=np.unicode_)
    data_splits = np.array([0, 2, 3], dtype=np.int32)
    separator = ""
    ngram_widths = [2]
    left_pad = ""
    right_pad = ""
    pad_width = 0
    preserve_short_sequences = False
    name = None
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array(["hello", "world", "tensorflow"], dtype=np.unicode_)
    data_splits = np.array([0, 1, 2, 3], dtype=np.int64)
    separator = "_"
    ngram_widths = [1, 2]
    left_pad = "<"
    right_pad = ">"
    pad_width = 1
    preserve_short_sequences = True
    name = "ngrams"
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array(["a", "b", "c", "d", "e"], dtype=np.unicode_)
    data_splits = np.array([0, 2, 5], dtype=np.int32)
    separator = "-"
    ngram_widths = [3]
    left_pad = "*"
    right_pad = "#"
    pad_width = -1
    preserve_short_sequences = False
    name = None
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array(["one", "two", "three", "four"], dtype=np.unicode_)
    data_splits = np.array([0, 1, 3, 4], dtype=np.int64)
    separator = " "
    ngram_widths = [2, 3]
    left_pad = "START"
    right_pad = "END"
    pad_width = 2
    preserve_short_sequences = True
    name = "ngrams2"
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array(["short", "longsentence"], dtype=np.unicode_)
    data_splits = np.array([0, 1, 2], dtype=np.int32)
    separator = "|"
    ngram_widths = [4]
    left_pad = "L"
    right_pad = "R"
    pad_width = 0
    preserve_short_sequences = True
    name = None
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array(["single"], dtype=np.unicode_)
    data_splits = np.array([0, 1], dtype=np.int64)
    separator = ":"
    ngram_widths = [1, 2, 3]
    left_pad = "BOS"
    right_pad = "EOS"
    pad_width = -1
    preserve_short_sequences = False
    name = "ngrams3"
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array(["", "empty", "string"], dtype=np.unicode_)
    data_splits = np.array([0, 1, 2, 3], dtype=np.int32)
    separator = "++"
    ngram_widths = [2]
    left_pad = "<<"
    right_pad = ">>"
    pad_width = 1
    preserve_short_sequences = True
    name = None
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array(["test", "data"], dtype=np.unicode_)
    data_splits = np.array([0, 2], dtype=np.int64)
    separator = "::"
    ngram_widths = [3,4]
    left_pad = "begin"
    right_pad = "end"
    pad_width = -1
    preserve_short_sequences = False
    name = "ngrams4"
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array(["a", "b", "c"], dtype=np.unicode_)
    data_splits = np.array([0, 3], dtype=np.int32)
    separator = "___"
    ngram_widths = [1]
    left_pad = "left"
    right_pad = "right"
    pad_width = 0
    preserve_short_sequences = True
    name = None
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array(["1", "2", "3", "4"], dtype=np.unicode_)
    data_splits = np.array([0, 2, 4], dtype=np.int64)
    separator = "---"
    ngram_widths = [2]
    left_pad = "start"
    right_pad = "finish"
    pad_width = -1
    preserve_short_sequences = False
    name = "ngrams5"
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    data = np.array(["very", "long", "string"], dtype=np.unicode_)
    data_splits = np.array([0, 1, 2, 3], dtype=np.int32)
    separator = "_"
    ngram_widths = [1, 2, 3]
    left_pad = "<s>"
    right_pad = "</s>"
    pad_width = 1
    preserve_short_sequences = True
    name = None
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    data = np.array(["hello"], dtype=np.unicode_)
    data_splits = np.array([0, 1], dtype=np.int32)
    separator = ""
    ngram_widths = [1, 2]
    left_pad = "<s>"
    right_pad = "</s>"
    pad_width = -1
    preserve_short_sequences = True
    name = "ngrams6"
    input_dict = {"data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringNGrams"] = tf_raw_ops_StringNGrams_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringNGrams' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringNGrams'.")

check_valid('tf.raw_ops.StringNGrams', generated_inputs['tf.raw_ops.StringNGrams'], lib="tf", suffix=0)
