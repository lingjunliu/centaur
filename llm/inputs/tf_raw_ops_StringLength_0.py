
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_stringlength_inputs():
    list_of_inputs = []

    # Input 1: Basic string tensor with BYTE unit
    input_tensor = np.array(["hello", "world", ""], dtype=object)
    unit = "BYTE"
    name = None
    input_dict = {"input": input_tensor, "unit": unit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String tensor with UTF8_CHAR unit
    input_tensor = np.array(["你好", "世界", ""], dtype=object)
    unit = "UTF8_CHAR"
    name = "string_length_utf8"
    input_dict = {"input": input_tensor, "unit": unit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string tensor
    input_tensor = np.array([], dtype=object)
    unit = "BYTE"
    name = None
    input_dict = {"input": input_tensor, "unit": unit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D string tensor with BYTE unit
    input_tensor = np.array([["hello", "world"], ["foo", "bar"]], dtype=object)
    unit = "BYTE"
    name = None
    input_dict = {"input": input_tensor, "unit": unit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D string tensor with UTF8_CHAR unit
    input_tensor = np.array([["你好", "世界"], ["你好世界", ""]], dtype=object)
    unit = "UTF8_CHAR"
    name = "string_length_utf8_2d"
    input_dict = {"input": input_tensor, "unit": unit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String tensor with mixed ASCII and UTF8 characters (BYTE)
    input_tensor = np.array(["hello你好", "world世界", ""], dtype=object)
    unit = "BYTE"
    name = None
    input_dict = {"input": input_tensor, "unit": unit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: String tensor with mixed ASCII and UTF8 characters (UTF8_CHAR)
    input_tensor = np.array(["hello你好", "world世界", ""], dtype=object)
    unit = "UTF8_CHAR"
    name = "string_length_mixed_utf8"
    input_dict = {"input": input_tensor, "unit": unit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String with special characters
    input_tensor = np.array(["!@#$%^&*()", "`-=[]\\;',./", "~_+{}|\":<>?"], dtype=object)
    unit = "BYTE"
    name = None
    input_dict = {"input": input_tensor, "unit": unit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Long strings
    long_string = "a" * 100
    input_tensor = np.array([long_string, "short"], dtype=object)
    unit = "BYTE"
    name = None
    input_dict = {"input": input_tensor, "unit": unit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Unicode characters
    input_tensor = np.array(["\U0001F600", "\U0001F642", "\U0001F680"], dtype=object)
    unit = "UTF8_CHAR"
    name = "string_length_unicode"
    input_dict = {"input": input_tensor, "unit": unit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for i in range(len(list_of_inputs)):
      list_of_inputs[i]['input'] = tf.constant(list_of_inputs[i]['input'])

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringLength"] = tf_raw_ops_stringlength_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringLength' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringLength'.")

check_valid('tf.raw_ops.StringLength', generated_inputs['tf.raw_ops.StringLength'], lib="tf", suffix=0)
