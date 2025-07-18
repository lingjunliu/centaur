
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_regex_replace_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar replacement, from documentation
    input_dict_1 = {
        'input': np.array("Text with tags.<br /><b>contains html</b>", dtype=object),
        'pattern': "<[^>]+>",
        'rewrite': " ",
        'replace_global': True,
        'name': 'basic_html_strip'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D tensor with multiple replacements per element
    input_dict_2 = {
        'input': np.array(["apple", "banana", "apricot"], dtype=object),
        'pattern': "a",
        'rewrite': "A",
        'replace_global': True,
        'name': 'vector_replace_all'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D tensor with replace_global=False
    input_dict_3 = {
        'input': np.array([["aa bb aa", "cc dd"], ["ee ff ee", "gg hh"]], dtype=object),
        'pattern': "[aeiou]",
        'rewrite': "V",
        'replace_global': False,
        'name': 'matrix_replace_first'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using capture groups to reorder
    input_dict_4 = {
        'input': np.array(["Doe, John", "Smith, Jane"], dtype=object),
        'pattern': r"(\w+), (\w+)",
        'rewrite': r"\2 \1",
        'replace_global': True,
        'name': 'capture_group_reorder'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Deleting characters by rewriting with an empty string
    input_dict_5 = {
        'input': np.array(["(555)-123-4567", "555.789.1234"], dtype=object),
        'pattern': r"[().-]",
        'rewrite': "",
        'replace_global': True,
        'name': 'delete_punctuation'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Pattern does not match
    input_dict_6 = {
        'input': np.array(["hello", "world"], dtype=object),
        'pattern': "xyz",
        'rewrite': "REPLACED",
        'replace_global': True,
        'name': 'no_match_found'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Input tensor with an empty string
    input_dict_7 = {
        'input': np.array(["first", "", "third"], dtype=object),
        'pattern': "i",
        'rewrite': "I",
        'replace_global': True,
        'name': 'input_with_empty_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Unicode characters in input
    input_dict_8 = {
        'input': np.array(["你好, 世界", "你好 Python"], dtype=object),
        'pattern': "你好",
        'rewrite': "Hello",
        'replace_global': True,
        'name': 'unicode_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Escaping special regex characters in the pattern
    input_dict_9 = {
        'input': np.array(["file.txt", "archive.zip", "doc."], dtype=object),
        'pattern': r"\.",
        'rewrite': "_",
        'replace_global': True,
        'name': 'escape_special_char'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: High-dimensional tensor input (3D)
    input_dict_10 = {
        'input': np.array([[["a1", "b2"], ["c3", "d4"]], [["e5", "f6"], ["g7", "h8"]]], dtype=object),
        'pattern': r'\d',
        'rewrite': '#',
        'replace_global': True,
        'name': '3d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Replacing with backslash characters in rewrite
    input_dict_11 = {
        'input': np.array(["path/to/file"], dtype=object),
        'pattern': "/",
        'rewrite': r"\\",
        'replace_global': True,
        'name': 'replace_with_backslash'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.strings.regex_replace"] = tf_strings_regex_replace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.regex_replace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.regex_replace'.")

check_valid('tf.strings.regex_replace', generated_inputs['tf.strings.regex_replace'], lib="tf", suffix=0)
