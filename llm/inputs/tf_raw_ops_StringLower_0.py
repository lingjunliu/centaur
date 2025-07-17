
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_stringlower_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array("CamelCase string and ALL CAPS", dtype=np.object_)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array("  Mixed CASE wITH sPaCeS  ", dtype=np.object_)
    encoding = "utf-8"
    name = "lower_case_op"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array("12345ABCdef", dtype=np.object_)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array("", dtype=np.object_)
    encoding = "utf-8"
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["HELLO", "WORLD"], dtype=np.object_)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([["HELLO", "WORLD"], ["AGAIN", "TEST"]], dtype=np.object_)
    encoding = "utf-8"
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array("!@#$%^&*()_+=-`~[]{}|;':\",./<>?", dtype=np.object_)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array(" HELLO WORLD", dtype=np.object_)
    encoding = "utf-8"
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array("This is a LONGER string with more words AND ALL CAPS", dtype=np.object_)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([b"HELLO", b"WORLD"], dtype=np.object_)
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringLower"] = tf_raw_ops_stringlower_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringLower' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringLower'.")

check_valid('tf.raw_ops.StringLower', generated_inputs['tf.raw_ops.StringLower'], lib="tf", suffix=0)
