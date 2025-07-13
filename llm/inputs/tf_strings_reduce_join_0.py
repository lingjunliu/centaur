
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_reduce_join_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.array([['abc','123'], ['def','456']], dtype=np.object_)
    axis = -1
    keepdims = False
    separator = " "
    name = "test1"
    input_dict = {"inputs": inputs, "axis": axis, "keepdims": keepdims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.array([['abc','123'], ['def','456']], dtype=np.object_)
    axis = None
    keepdims = False
    separator = ""
    name = None
    input_dict = {"inputs": inputs, "axis": axis, "keepdims": keepdims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.array([[['abc','123'], ['def','456']], [['ghi','789'], ['jkl','012']]], dtype=np.object_)
    axis = 0
    keepdims = False
    separator = "-"
    name = "test3"
    input_dict = {"inputs": inputs, "axis": axis, "keepdims": keepdims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.array([[['abc','123'], ['def','456']], [['ghi','789'], ['jkl','012']]], dtype=np.object_)
    axis = 1
    keepdims = True
    separator = ","
    name = None
    input_dict = {"inputs": inputs, "axis": axis, "keepdims": keepdims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.array([['a','b','c'], ['d','e','f']], dtype=np.object_)
    axis = 1
    keepdims = False
    separator = "---"
    name = "test5"
    input_dict = {"inputs": inputs, "axis": axis, "keepdims": keepdims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.array([['hello'], ['world']], dtype=np.object_)
    axis = 0
    keepdims = False
    separator = "\n"
    name = None
    input_dict = {"inputs": inputs, "axis": axis, "keepdims": keepdims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.array([['1', '2'], ['3', '4']], dtype=np.object_)
    axis = -1
    keepdims = True
    separator = "+"
    name = "test7"
    input_dict = {"inputs": inputs, "axis": axis, "keepdims": keepdims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = np.array([['one', 'two', 'three']], dtype=np.object_)
    axis = None
    keepdims = False
    separator = " "
    name = None
    input_dict = {"inputs": inputs, "axis": axis, "keepdims": keepdims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.array([['', ''], ['', '']], dtype=np.object_)
    axis = 1
    keepdims = False
    separator = "x"
    name = "test9"
    input_dict = {"inputs": inputs, "axis": axis, "keepdims": keepdims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.array([['a']], dtype=np.object_)
    axis = None
    keepdims = False
    separator = ""
    name = None
    input_dict = {"inputs": inputs, "axis": axis, "keepdims": keepdims, "separator": separator, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.reduce_join"] = tf_strings_reduce_join_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.reduce_join' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.reduce_join'.")

check_valid('tf.strings.reduce_join', generated_inputs['tf.strings.reduce_join'], lib="tf", suffix=0)
