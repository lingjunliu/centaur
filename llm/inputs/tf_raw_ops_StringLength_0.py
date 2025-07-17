
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_StringLength_inputs():
    list_of_inputs = []

    # Input 1: Basic string tensor, byte length
    input_tensor = tf.convert_to_tensor(np.array(["hello", "world", ""], dtype=np.object_))
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String tensor, UTF8 character length
    input_tensor = tf.convert_to_tensor(np.array(["你好", "世界", ""], dtype=np.object_))
    input_dict = {"input": input_tensor, "unit": "UTF8_CHAR", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string tensor
    input_tensor = tf.convert_to_tensor(np.array([""], dtype=np.object_))
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": "empty_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String tensor with special characters, byte length
    input_tensor = tf.convert_to_tensor(np.array(["!@#$%^", "&*()_+", "-=[]{}|;':", ",./<>?"], dtype=np.object_))
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String tensor with special characters, UTF8 length (all ASCII)
    input_tensor = tf.convert_to_tensor(np.array(["!@#$%^", "&*()_+", "-=[]{}|;':", ",./<>?"], dtype=np.object_))
    input_dict = {"input": input_tensor, "unit": "UTF8_CHAR", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String tensor with mixed ASCII and UTF8 characters, byte length
    input_tensor = tf.convert_to_tensor(np.array(["hello你好", "world世界", ""], dtype=np.object_))
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: String tensor with mixed ASCII and UTF8 characters, UTF8 length
    input_tensor = tf.convert_to_tensor(np.array(["hello你好", "world世界", ""], dtype=np.object_))
    input_dict = {"input": input_tensor, "unit": "UTF8_CHAR", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  byte length
    input_tensor = tf.convert_to_tensor(np.array([["hello"]], dtype=np.object_))
    input_dict = {"input": input_tensor, "unit": "BYTE", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: UTF8 length
    input_tensor = tf.convert_to_tensor(np.array([["你好"]], dtype=np.object_))
    input_dict = {"input": input_tensor, "unit": "UTF8_CHAR", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with empty strings and UTF8 chars, UTF8 length
    input_tensor = tf.convert_to_tensor(np.array(["", "世界", ""], dtype=np.object_))
    input_dict = {"input": input_tensor, "unit": "UTF8_CHAR", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringLength"] = tf_raw_ops_StringLength_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringLength' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringLength'.")

check_valid('tf.raw_ops.StringLength', generated_inputs['tf.raw_ops.StringLength'], lib="tf", suffix=0)
