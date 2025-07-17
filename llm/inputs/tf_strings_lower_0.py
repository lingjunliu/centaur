
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_lower_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = tf.constant("CamelCase string and ALL CAPS").numpy()
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = tf.constant("MixedCase123!@#").numpy()
    encoding = "utf-8"
    name = "lower_case_string"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = tf.constant(["String1", "String2", "String3"]).numpy()
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = tf.constant([["String1", "String2"], ["String3", "String4"]]).numpy()
    encoding = "utf-8"
    name = "name2"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = tf.constant("ALREADY LOWER").numpy()
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = tf.constant("Numbers12345").numpy()
    encoding = "utf-8"
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = tf.constant("").numpy()
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = tf.constant("   Leading and Trailing Spaces   ").numpy()
    encoding = "utf-8"
    name = "space_test"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = tf.constant("!@#$%^&*()").numpy()
    encoding = ""
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([s.decode('utf-8') for s in [b"BYtestring1", b"BYtestring2"]], dtype=np.str_)
    encoding = "utf-8"
    name = None
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.lower"] = tf_strings_lower_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.lower' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.lower'.")

check_valid('tf.strings.lower', generated_inputs['tf.strings.lower'], lib="tf", suffix=0)
