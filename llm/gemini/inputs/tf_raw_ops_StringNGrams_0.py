
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_stringngrams_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.StringNGrams function.
    """
    list_of_inputs = []

    # Input 1: Basic bigrams with a space separator.
    input_dict_1 = {
        'name': 'basic_bigrams',
        'data': np.array(['a', 'b', 'c', 'd', 'e'], dtype=np.object_),
        'data_splits': np.array([0, 3, 5], dtype=np.int32),
        'separator': ' ',
        'ngram_widths': [2],
        'left_pad': '',
        'right_pad': '',
        'pad_width': 0,
        'preserve_short_sequences': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Trigrams with no separator.
    input_dict_2 = {
        'name': 'trigrams_no_sep',
        'data': np.array(['one', 'two', 'three', 'four', 'five'], dtype=np.object_),
        'data_splits': np.array([0, 5], dtype=np.int64),
        'separator': '',
        'ngram_widths': [3],
        'left_pad': '',
        'right_pad': '',
        'pad_width': 0,
        'preserve_short_sequences': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Multiple ngram widths (bigrams and trigrams).
    input_dict_3 = {
        'name': 'multi_width_ngrams',
        'data': np.array(['a', 'b', 'c', 'd'], dtype=np.object_),
        'data_splits': np.array([0, 4], dtype=np.int32),
        'separator': '_',
        'ngram_widths': [2, 3],
        'left_pad': '',
        'right_pad': '',
        'pad_width': 0,
        'preserve_short_sequences': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Padding enabled with custom pad strings.
    input_dict_4 = {
        'name': 'simple_padding',
        'data': np.array(['a', 'b', 'c'], dtype=np.object_),
        'data_splits': np.array([0, 3], dtype=np.int32),
        'separator': ' ',
        'ngram_widths': [3],
        'left_pad': 'LP',
        'right_pad': 'RP',
        'pad_width': 1,
        'preserve_short_sequences': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Max padding (pad_width = -1).
    input_dict_5 = {
        'name': 'max_padding',
        'data': np.array(['a', 'b', 'c'], dtype=np.object_),
        'data_splits': np.array([0, 1, 3], dtype=np.int64),
        'separator': '+',
        'ngram_widths': [2, 3],
        'left_pad': '<S>',
        'right_pad': '</S>',
        'pad_width': -1,
        'preserve_short_sequences': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: preserve_short_sequences=False, where one sequence is shorter than ngram_width.
    input_dict_6 = {
        'name': 'preserve_false',
        'data': np.array(['x', 'y', 'z', 'w'], dtype=np.object_),
        'data_splits': np.array([0, 1, 4], dtype=np.int32),
        'separator': ' ',
        'ngram_widths': [2],
        'left_pad': '',
        'right_pad': '',
        'pad_width': 0,
        'preserve_short_sequences': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: preserve_short_sequences=True, where one sequence is shorter than ngram_width.
    input_dict_7 = {
        'name': 'preserve_true',
        'data': np.array(['x', 'y', 'z', 'w'], dtype=np.object_),
        'data_splits': np.array([0, 1, 4], dtype=np.int32),
        'separator': ' ',
        'ngram_widths': [2],
        'left_pad': '',
        'right_pad': '',
        'pad_width': 0,
        'preserve_short_sequences': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty input data tensor.
    input_dict_8 = {
        'name': 'empty_input',
        'data': np.array([], dtype=np.object_),
        'data_splits': np.array([0], dtype=np.int32),
        'separator': ' ',
        'ngram_widths': [2],
        'left_pad': 'LP',
        'right_pad': 'RP',
        'pad_width': 1,
        'preserve_short_sequences': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Padding width that will be clipped (pad_width > ngram_widths-1).
    input_dict_9 = {
        'name': 'clipped_padding',
        'data': np.array(['a', 'b'], dtype=np.object_),
        'data_splits': np.array([0, 2], dtype=np.int32),
        'separator': ' ',
        'ngram_widths': [2],
        'left_pad': 'P',
        'right_pad': 'P',
        'pad_width': 5,
        'preserve_short_sequences': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Unigrams (ngram_widths=[1]).
    input_dict_10 = {
        'name': 'unigrams',
        'data': np.array(['a', 'b', 'c'], dtype=np.object_),
        'data_splits': np.array([0, 1, 2, 3], dtype=np.int64),
        'separator': ',',
        'ngram_widths': [1],
        'left_pad': '',
        'right_pad': '',
        'pad_width': 0,
        'preserve_short_sequences': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Unicode characters in data and pads.
    input_dict_11 = {
        'name': 'unicode_input',
        'data': np.array(['你好', '👋', 'world', '!', '👍'], dtype=np.object_),
        'data_splits': np.array([0, 5], dtype=np.int32),
        'separator': '_',
        'ngram_widths': [2, 3],
        'left_pad': 'START✅',
        'right_pad': 'END🛑',
        'pad_width': -1,
        'preserve_short_sequences': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Empty sequence within the ragged tensor.
    input_dict_12 = {
        'name': 'empty_sequence',
        'data': np.array(['a', 'b', 'c'], dtype=np.object_),
        'data_splits': np.array([0, 2, 2, 3], dtype=np.int32), # ['a','b'], [], ['c']
        'separator': ' ',
        'ngram_widths': [2],
        'left_pad': 'L',
        'right_pad': 'R',
        'pad_width': 1,
        'preserve_short_sequences': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.StringNGrams"] = tf_raw_ops_stringngrams_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.StringNGrams' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringNGrams'.")

check_valid('tf.raw_ops.StringNGrams', generated_inputs['tf.raw_ops.StringNGrams'], lib="tf", suffix=0)
