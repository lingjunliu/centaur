
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unsorted_segment_join_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.array(['this', 'a', 'test', 'is'], dtype=object)
    segment_ids = np.array([0, 1, 1, 0], dtype=np.int32)
    num_segments = 2
    separator = ' '
    name = None
    input_dict = {'inputs': inputs, 'segment_ids': segment_ids, 'num_segments': num_segments, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.array([['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']], dtype=object)
    segment_ids = np.array([1, 0, 1], dtype=np.int32)
    num_segments = 2
    separator = ':'
    name = None
    input_dict = {'inputs': inputs, 'segment_ids': segment_ids, 'num_segments': num_segments, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.array(['hello', 'world', 'tensorflow'], dtype=object)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    num_segments = 2
    separator = ''
    name = None
    input_dict = {'inputs': inputs, 'segment_ids': segment_ids, 'num_segments': num_segments, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.array([['a', 'b'], ['c', 'd'], ['e', 'f']], dtype=object)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = 2
    separator = ','
    name = None
    input_dict = {'inputs': inputs, 'segment_ids': segment_ids, 'num_segments': num_segments, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.array(['1', '2', '3', '4', '5'], dtype=object)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    num_segments = 3
    separator = '-'
    name = None
    input_dict = {'inputs': inputs, 'segment_ids': segment_ids, 'num_segments': num_segments, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.array([['x', 'y'], ['z', 'w'], ['u', 'v'], ['s', 't']], dtype=object)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    num_segments = 2
    separator = '|'
    name = "test_segment_join"
    input_dict = {'inputs': inputs, 'segment_ids': segment_ids, 'num_segments': num_segments, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.array(['apple', 'banana', 'cherry', 'date'], dtype=object)
    segment_ids = np.array([3, 1, 2, 0], dtype=np.int32)
    num_segments = 4
    separator = ';'
    name = None
    input_dict = {'inputs': inputs, 'segment_ids': segment_ids, 'num_segments': num_segments, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = np.array([['one', 'two'], ['three', 'four'], ['five', 'six']], dtype=object)
    segment_ids = np.array([2, 0, 1], dtype=np.int64)
    num_segments = 3
    separator = '::'
    name = None
    input_dict = {'inputs': inputs, 'segment_ids': segment_ids, 'num_segments': num_segments, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.array(['A', 'B', 'C', 'D', 'E', 'F'], dtype=object)
    segment_ids = np.array([5, 4, 3, 2, 1, 0], dtype=np.int32)
    num_segments = 6
    separator = '_'
    name = None
    input_dict = {'inputs': inputs, 'segment_ids': segment_ids, 'num_segments': num_segments, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.array([['1', '1'], ['2', '2'], ['3', '3'], ['4', '4']], dtype=object)
    segment_ids = np.array([0, 1, 0, 1], dtype=np.int32)
    num_segments = 2
    separator = "---"
    name = "another_test"
    input_dict = {'inputs': inputs, 'segment_ids': segment_ids, 'num_segments': num_segments, 'separator': separator, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.unsorted_segment_join"] = tf_strings_unsorted_segment_join_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.unsorted_segment_join' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unsorted_segment_join'.")

check_valid('tf.strings.unsorted_segment_join', generated_inputs['tf.strings.unsorted_segment_join'], lib="tf", suffix=0)
