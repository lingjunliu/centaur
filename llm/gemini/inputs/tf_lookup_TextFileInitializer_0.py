
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_lookup_TextFileInitializer_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'filename': 'test.txt',
        'key_dtype': np.int64,
        'key_index': 0,
        'value_dtype': np.int64,
        'value_index': 1,
        'vocab_size': 10,
        'delimiter': ',',
        'name': 'initializer1',
        'value_index_offset': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'filename': 'data.txt',
        'key_dtype': np.string_,
        'key_index': 2,
        'value_dtype': np.int64,
        'value_index': 0,
        'vocab_size': 5,
        'delimiter': ' ',
        'name': 'initializer2',
        'value_index_offset': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'filename': 'words.txt',
        'key_dtype': np.string_,
        'key_index': 0,
        'value_dtype': np.int64,
        'value_index': tf.lookup.TextFileIndex.LINE_NUMBER,
        'vocab_size': 20,
        'delimiter': '\t',
        'name': 'initializer3',
        'value_index_offset': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'filename': 'sentences.txt',
        'key_dtype': np.string_,
        'key_index': tf.lookup.TextFileIndex.WHOLE_LINE,
        'value_dtype': np.int64,
        'value_index': 0,
        'vocab_size': 15,
        'delimiter': '|',
        'name': 'initializer4',
        'value_index_offset': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    input_dict = {
        'filename': 'numbers.txt',
        'key_dtype': np.int64,
        'key_index': 1,
        'value_dtype': np.int64,
        'value_index': 2,
        'vocab_size': None,
        'delimiter': ' ',
        'name': None,
        'value_index_offset': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'filename': 'mixed.txt',
        'key_dtype': np.string_,
        'key_index': 3,
        'value_dtype': np.int64,
        'value_index': 1,
        'vocab_size': None,
        'delimiter': ',',
        'name': 'mixed_data',
        'value_index_offset': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'filename': 'another_test.txt',
        'key_dtype': np.int32,
        'key_index': 0,
        'value_dtype': np.int64,
        'value_index': 1,
        'vocab_size': 100,
        'delimiter': ';',
        'name': 'rev_init',
        'value_index_offset': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'filename': 'more_data.txt',
        'key_dtype': np.float32,
        'key_index': 2,
        'value_dtype': np.int64,
        'value_index': 1,
        'vocab_size': 25,
        'delimiter': ':',
        'name': 'numbers_init',
        'value_index_offset': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'filename': 'last_one.txt',
        'key_dtype': np.string_,
        'key_index': tf.lookup.TextFileIndex.WHOLE_LINE,
        'value_dtype': np.int64,
        'value_index': tf.lookup.TextFileIndex.LINE_NUMBER,
        'vocab_size': 50,
        'delimiter': '\n',
        'name': 'whole_line_init',
        'value_index_offset': -10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'filename': 'final.txt',
        'key_dtype': np.int64,
        'key_index': 1,
        'value_dtype': np.int64,
        'value_index': 0,
        'vocab_size': None,
        'delimiter': ',',
        'name': None,
        'value_index_offset': 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.lookup.TextFileInitializer"] = tf_lookup_TextFileInitializer_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.lookup.TextFileInitializer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.lookup.TextFileInitializer'.")

check_valid('tf.lookup.TextFileInitializer', generated_inputs['tf.lookup.TextFileInitializer'], lib="tf", suffix=0)
