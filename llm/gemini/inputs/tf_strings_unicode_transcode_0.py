
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_transcode_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["Hello", "TensorFlow", "2.x"], dtype=np.object_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-16-BE"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "output_encoding": output_encoding,
                  "errors": errors, "replacement_char": replacement_char,
                  "replace_control_characters": replace_control_characters, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["A", "B", "C"], dtype=np.object_)
    input_encoding = "US ASCII"
    output_encoding = "UTF-8"
    errors = "strict"
    replacement_char = 65533
    replace_control_characters = False
    name = "transcode_op"
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "output_encoding": output_encoding,
                  "errors": errors, "replacement_char": replacement_char,
                  "replace_control_characters": replace_control_characters, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["你好", "世界"], dtype=np.object_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-32-BE"
    errors = "ignore"
    replacement_char = 0
    replace_control_characters = False
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "output_encoding": output_encoding,
                  "errors": errors, "replacement_char": replacement_char,
                  "replace_control_characters": replace_control_characters, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(["123", "456", "789"], dtype=np.object_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 32
    replace_control_characters = False
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "output_encoding": output_encoding,
                  "errors": errors, "replacement_char": replacement_char,
                  "replace_control_characters": replace_control_characters, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["\x01\x02\x03", "\x04\x05\x06"], dtype=np.object_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "output_encoding": output_encoding,
                  "errors": errors, "replacement_char": replacement_char,
                  "replace_control_characters": replace_control_characters, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array(["Test"], dtype=np.object_)
    input_encoding = "UTF-16"
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "output_encoding": output_encoding,
                  "errors": errors, "replacement_char": replacement_char,
                  "replace_control_characters": replace_control_characters, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array(["Invalid UTF-8: \xFF"], dtype=np.object_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "output_encoding": output_encoding,
                  "errors": errors, "replacement_char": replacement_char,
                  "replace_control_characters": replace_control_characters, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multidimensional tensor
    input_tensor = np.array([["Hello", "World"], ["TensorFlow", "Keras"]], dtype=np.object_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-16-BE"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "output_encoding": output_encoding,
                  "errors": errors, "replacement_char": replacement_char,
                  "replace_control_characters": replace_control_characters, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty strings
    input_tensor = np.array(["", ""], dtype=np.object_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    replace_control_characters = False
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "output_encoding": output_encoding,
                  "errors": errors, "replacement_char": replacement_char,
                  "replace_control_characters": replace_control_characters, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Using a different replacement char
    input_tensor = np.array(["Invalid UTF-8: \xFF"], dtype=np.object_)
    input_encoding = "UTF-8"
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 63  # ASCII code for '?'
    replace_control_characters = False
    name = None
    input_dict = {"input": input_tensor, "input_encoding": input_encoding, "output_encoding": output_encoding,
                  "errors": errors, "replacement_char": replacement_char,
                  "replace_control_characters": replace_control_characters, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.unicode_transcode"] = tf_strings_unicode_transcode_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.unicode_transcode' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_transcode'.")

check_valid('tf.strings.unicode_transcode', generated_inputs['tf.strings.unicode_transcode'], lib="tf", suffix=0)
