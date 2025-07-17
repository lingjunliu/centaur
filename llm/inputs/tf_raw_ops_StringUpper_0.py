
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_upper_inputs():
    list_of_inputs = []

    # Helper function to create a tf.string tensor from a numpy array
    def create_string_tensor(data):
        if isinstance(data, str):
            data = np.array(data, dtype=np.object_)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.object_)
        return tf.constant(data)

    # Input 1: Basic string
    input_tensor = create_string_tensor("hello world")
    encoding = ""
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String with special characters
    input_tensor = create_string_tensor("123 abc!@#")
    encoding = ""
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string
    input_tensor = create_string_tensor("")
    encoding = ""
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: UTF-8 encoding
    input_tensor = create_string_tensor("héllo world")
    encoding = "utf-8"
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String with mixed case
    input_tensor = create_string_tensor("HeLlO wOrLd")
    encoding = ""
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String with numbers and uppercase
    input_tensor = create_string_tensor("123ABC")
    encoding = ""
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: String with only uppercase letters
    input_tensor = create_string_tensor("UPPERCASE")
    encoding = ""
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String with spaces only
    input_tensor = create_string_tensor("   ")
    encoding = ""
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String with multiple lines
    input_tensor = create_string_tensor("line1\nline2")
    encoding = ""
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String with Unicode characters
    input_tensor = create_string_tensor("你好世界")
    encoding = "utf-8"
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Name is specified
    input_tensor = create_string_tensor("hello world")
    encoding = ""
    name = "my_op"
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Tensor of shape (2, 2)
    input_tensor = create_string_tensor([["hello", "world"], ["foo", "bar"]])
    encoding = ""
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Tensor of shape (1,)
    input_tensor = create_string_tensor(["hello"])
    encoding = ""
    name = ""
    input_dict = {"input": input_tensor, "encoding": encoding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringUpper"] = tf_raw_ops_string_upper_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringUpper' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringUpper'.")

check_valid('tf.raw_ops.StringUpper', generated_inputs['tf.raw_ops.StringUpper'], lib="tf", suffix=0)
