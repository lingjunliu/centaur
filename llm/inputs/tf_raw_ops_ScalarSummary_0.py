
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ScalarSummary_inputs():
    list_of_inputs = []

    # Input 1
    tags = np.array(['tag1', 'tag2'], dtype=np.str_)
    values = np.array([1.0, 2.0], dtype=np.float32)
    name = "summary1"
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tags = np.array(['tag3'], dtype=np.str_)
    values = np.array([3.14159], dtype=np.float64)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tags = np.array(['tag4', 'tag5', 'tag6'], dtype=np.str_)
    values = np.array([10, 20, 30], dtype=np.int32)
    name = "summary3"
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tags = np.array(['tag7', 'tag8'], dtype=np.str_)
    values = np.array([255, 128], dtype=np.uint8)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tags = np.array(['tag9'], dtype=np.str_)
    values = np.array([-1000], dtype=np.int16)
    name = "summary5"
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tags = np.array(['tag10', 'tag11'], dtype=np.str_)
    values = np.array([-127, 127], dtype=np.int8)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tags = np.array(['tag12'], dtype=np.str_)
    values = np.array([10000000000], dtype=np.int64)
    name = "summary7"
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tags = np.array(['tag13', 'tag14'], dtype=np.str_)
    values = np.array([1.0, 2.0], dtype=np.float32)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tags = np.array(['tag15'], dtype=np.str_)
    values = np.array([100], dtype=np.int32)
    name = "summary9"
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tags = np.array(['tag16', 'tag17'], dtype=np.str_)
    values = np.array([1.0, 2.0], dtype=np.float32)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScalarSummary"] = tf_raw_ops_ScalarSummary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScalarSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScalarSummary'.")

check_valid('tf.raw_ops.ScalarSummary', generated_inputs['tf.raw_ops.ScalarSummary'], lib="tf", suffix=0)
