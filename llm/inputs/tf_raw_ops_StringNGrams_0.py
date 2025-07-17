
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_ngrams_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([b"hello", b"world", b"tensorflow", b"rocks"], dtype=np.object_)
    data_splits = np.array([0, 2, 4], dtype=np.int32)
    separator = b""
    ngram_widths = [2]
    left_pad = b""
    right_pad = b""
    pad_width = 0
    preserve_short_sequences = False
    name = "ngrams_1"
    input_dict = {"name": name, "data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([b"a", b"b", b"c", b"d", b"e", b"f"], dtype=np.object_)
    data_splits = np.array([0, 3, 6], dtype=np.int64)
    separator = b"_"
    ngram_widths = [2, 3]
    left_pad = b"<PAD>"
    right_pad = b"</PAD>"
    pad_width = 1
    preserve_short_sequences = True
    name = "ngrams_2"
    input_dict = {"name": name, "data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([b"one", b"two", b"three"], dtype=np.object_)
    data_splits = np.array([0, 1, 2, 3], dtype=np.int32)
    separator = b"-"
    ngram_widths = [1, 2, 3]
    left_pad = b"[START]"
    right_pad = b"[END]"
    pad_width = -1
    preserve_short_sequences = False
    name = "ngrams_3"
    input_dict = {"name": name, "data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([b"short", b"sentence"], dtype=np.object_)
    data_splits = np.array([0, 2], dtype=np.int64)
    separator = b" "
    ngram_widths = [2]
    left_pad = b"BEGIN"
    right_pad = b"END"
    pad_width = 2
    preserve_short_sequences = True
    name = "ngrams_4"
    input_dict = {"name": name, "data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([b"very", b"long", b"string"], dtype=np.object_)
    data_splits = np.array([0, 1, 3], dtype=np.int32)
    separator = b"|"
    ngram_widths = [3]
    left_pad = b"<<<<"
    right_pad = b">>>>"
    pad_width = 3
    preserve_short_sequences = False
    name = "ngrams_5"
    input_dict = {"name": name, "data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([b"test", b"case"], dtype=np.object_)
    data_splits = np.array([0, 1, 2], dtype=np.int64)
    separator = b","
    ngram_widths = [1]
    left_pad = b""
    right_pad = b""
    pad_width = 0
    preserve_short_sequences = True
    name = "ngrams_6"
    input_dict = {"name": name, "data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([b"example"], dtype=np.object_)
    data_splits = np.array([0, 1], dtype=np.int32)
    separator = b"++"
    ngram_widths = [4]
    left_pad = b"start"
    right_pad = b"end"
    pad_width = -1
    preserve_short_sequences = False
    name = "ngrams_7"
    input_dict = {"name": name, "data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array([b"more", b"data"], dtype=np.object_)
    data_splits = np.array([0, 1, 2], dtype=np.int64)
    separator = b"::"
    ngram_widths = [2, 4]
    left_pad = b"left"
    right_pad = b"right"
    pad_width = 1
    preserve_short_sequences = True
    name = "ngrams_8"
    input_dict = {"name": name, "data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array([b"different", b"widths"], dtype=np.object_)
    data_splits = np.array([0, 2], dtype=np.int32)
    separator = b"***"
    ngram_widths = [3, 5]
    left_pad = b"lpad"
    right_pad = b"rpad"
    pad_width = 2
    preserve_short_sequences = False
    name = "ngrams_9"
    input_dict = {"name": name, "data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([b"last", b"example"], dtype=np.object_)
    data_splits = np.array([0, 1, 2], dtype=np.int64)
    separator = b"###"
    ngram_widths = [1, 2, 3, 4]
    left_pad = b"lp"
    right_pad = b"rp"
    pad_width = -1
    preserve_short_sequences = True
    name = "ngrams_10"
    input_dict = {"name": name, "data": data, "data_splits": data_splits, "separator": separator, "ngram_widths": ngram_widths, "left_pad": left_pad, "right_pad": right_pad, "pad_width": pad_width, "preserve_short_sequences": preserve_short_sequences}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringNGrams"] = tf_raw_ops_string_ngrams_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringNGrams' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringNGrams'.")

check_valid('tf.raw_ops.StringNGrams', generated_inputs['tf.raw_ops.StringNGrams'], lib="tf", suffix=0)
