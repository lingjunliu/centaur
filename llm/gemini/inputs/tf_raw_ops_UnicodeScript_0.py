
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_UnicodeScript_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    input1 = np.array([1, 31, 38], dtype=np.int32)
    input_dict1 = {"input": input1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D array
    input2 = np.array([[65, 66], [67, 68]], dtype=np.int32)
    input_dict2 = {"input": input2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D array
    input3 = np.array([[[70, 71], [72, 73]], [[74, 75], [76, 77]]], dtype=np.int32)
    input_dict3 = {"input": input3, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Array with some invalid code points (-1)
    input4 = np.array([65, -1, 67, -1], dtype=np.int32)
    input_dict4 = {"input": input4, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Array with zero
    input5 = np.array([0, 66, 0, 68], dtype=np.int32)
    input_dict5 = {"input": input5, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Larger Unicode values
    input6 = np.array([65535, 65536, 65537], dtype=np.int32)
    input_dict6 = {"input": input6, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Another 2D array
    input7 = np.array([[1040, 1041], [1042, 1043]], dtype=np.int32)
    input_dict7 = {"input": input7, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Array with a name
    input8 = np.array([48, 49, 50, 51], dtype=np.int32)
    input_dict8 = {"input": input8, "name": "my_unicode_script"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Empty array
    input9 = np.array([], dtype=np.int32)
    input_dict9 = {"input": input9, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: Array with a mix of ASCII and larger unicode
    input10 = np.array([65, 128175, 66], dtype=np.int32)
    input_dict10 = {"input": input10, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
