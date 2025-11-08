
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: Basic string content
    filename = np.array("test_file.txt", dtype=np.string_)
    contents = np.array("Hello World!", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: File with newline characters
    filename = np.array("file_with_newlines.txt", dtype=np.string_)
    contents = np.array("Line1\nLine2\nLine3", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: File with special characters
    filename = np.array("special_chars.txt", dtype=np.string_)
    contents = np.array("Hello@#$%^&*()", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Empty content file
    filename = np.array("empty_file.txt", dtype=np.string_)
    contents = np.array("", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: File with numbers
    filename = np.array("numbers_file.txt", dtype=np.string_)
    contents = np.array("123456789", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: File with spaces
    filename = np.array("spaces_file.txt", dtype=np.string_)
    contents = np.array("   ", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: File with multiple lines
    filename = np.array("multi_line_file.txt", dtype=np.string_)
    contents = np.array("Line1\nLine2\nLine3\nLine4", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: File with tab characters
    filename = np.array("tab_file.txt", dtype=np.string_)
    contents = np.array("Tab\tCharacter\tHere", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: File with mixed content
    filename = np.array("mixed_file.txt", dtype=np.string_)
    contents = np.array("Mixed content\nWith\nNumbers: 123456789", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: File with binary content
    filename = np.array("binary_file.bin", dtype=np.string_)
    contents = np.array(b"\x00\x01\x02\x03", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.WriteFile"] = generate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.WriteFile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.WriteFile'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.WriteFile', generated_inputs['tf.raw_ops.WriteFile'], lib="tf", suffix=0)
