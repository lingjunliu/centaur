
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_length_inputs():
    list_of_inputs = []

    # Input 1: Basic string tensor, BYTE unit
    input_tensor = np.array(["hello", "world", ""], dtype=np.object_)
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String tensor with UTF-8 characters, UTF8_CHAR unit
    input_tensor = np.array(["你好", "世界", "你好世界"], dtype=np.object_)
    input_dict = {"input": input_tensor, "unit": "UTF8_CHAR", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string tensor, BYTE unit
    input_tensor = np.array(["", "", ""], dtype=np.object_)
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String tensor with special characters, BYTE unit
    input_tensor = np.array(["!@#$", "%^&*", "()_+"], dtype=np.object_)
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String tensor with mixed characters, UTF8_CHAR unit
    input_tensor = np.array(["hello你好", "world世界", "mixed你好世界"], dtype=np.object_)
    input_dict = {"input": input_tensor, "unit": "UTF8_CHAR", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional string tensor, BYTE unit
    input_tensor = np.array([["hello", "world"], ["你好", "世界"]], dtype=np.object_)
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multidimensional string tensor, UTF8_CHAR unit
    input_tensor = np.array([["hello", "world"], ["你好", "世界"]], dtype=np.object_)
    input_dict = {"input": input_tensor, "unit": "UTF8_CHAR", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String tensor with emojis, UTF8_CHAR unit
    input_tensor = np.array(["😀", "😂", "🤣"], dtype=np.object_)
    input_dict = {"input": input_tensor, "unit": "UTF8_CHAR", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String tensor with numbers, BYTE unit
    input_tensor = np.array(["123", "456", "789"], dtype=np.object_)
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String tensor with escape sequences, BYTE unit
    input_tensor = np.array(["\\n", "\\t", "\\r"], dtype=np.object_)
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.length"] = tf_strings_length_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.length' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.length'.")

check_valid('tf.strings.length', generated_inputs['tf.strings.length'], lib="tf", suffix=0)
