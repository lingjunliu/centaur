
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_UnicodeScript_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    input_tensor = np.array([1, 31, 38], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array
    input_tensor = np.array([[65, 66], [67, 68]], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array
    input_tensor = np.array([[[70, 71], [72, 73]], [[74, 75], [76, 77]]], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Array with some invalid code points
    input_tensor = np.array([1, 31, -1, 1114111, 38, 1114112], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with Unicode characters from different scripts.
    input_tensor = np.array([0x0041, 0x0628, 0x3042, 0x0410], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.UnicodeScript"] = tf_raw_ops_UnicodeScript_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.UnicodeScript' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnicodeScript'.")

check_valid('tf.raw_ops.UnicodeScript', generated_inputs['tf.raw_ops.UnicodeScript'], lib="tf", suffix=0)
