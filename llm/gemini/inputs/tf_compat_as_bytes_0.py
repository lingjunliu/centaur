
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_compat_as_bytes_inputs():
    list_of_inputs = []

    # Input 1
    bytes_or_text = "hello"
    encoding = "utf-8"
    input_dict = {"bytes_or_text": bytes_or_text, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    bytes_or_text = "你好"
    encoding = "utf-8"
    input_dict = {"bytes_or_text": bytes_or_text, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    bytes_or_text = "This is a test string with some special characters: !@#$%^&*()"
    encoding = "utf-8"
    input_dict = {"bytes_or_text": bytes_or_text, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    bytes_or_text = "1234567890"
    encoding = "utf-8"
    input_dict = {"bytes_or_text": bytes_or_text, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    bytes_or_text = "abcABC123"
    encoding = "utf-8"
    input_dict = {"bytes_or_text": bytes_or_text, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    bytes_or_text = "こんにちは"
    encoding = "utf-8"
    input_dict = {"bytes_or_text": bytes_or_text, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    bytes_or_text = ""
    encoding = "utf-8"
    input_dict = {"bytes_or_text": bytes_or_text, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    bytes_or_text = "Test with different encoding chars"
    encoding = "ascii"
    input_dict = {"bytes_or_text": bytes_or_text, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    bytes_or_text = "Testing more characters: äöüß"
    encoding = "utf-8"
    input_dict = {"bytes_or_text": bytes_or_text, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    bytes_or_text = "Some more text to test"
    encoding = "latin-1"
    input_dict = {"bytes_or_text": bytes_or_text, "encoding": encoding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.compat.as_bytes"] = tf_compat_as_bytes_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.compat.as_bytes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.as_bytes'.")

check_valid('tf.compat.as_bytes', generated_inputs['tf.compat.as_bytes'], lib="tf", suffix=0)
