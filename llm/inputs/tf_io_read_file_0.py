
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os

def tf_io_read_file_inputs():
    list_of_inputs = []

    # Input 1: Simple text file
    filename = "test_file_1.txt"
    with open(filename, "w") as f:
        f.write("Hello, world!")
    input_dict = {"filename": filename, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #os.remove(filename) # Removed os.remove because file should exist when tf reads it

    # Input 2: Empty file
    filename = "test_file_2.txt"
    open(filename, "w").close()
    input_dict = {"filename": filename, "name": "empty_file"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #os.remove(filename)

    # Input 3: File with numbers
    filename = "test_file_3.txt"
    with open(filename, "w") as f:
        f.write("1234567890")
    input_dict = {"filename": filename, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #os.remove(filename)

    # Input 4: File with special characters
    filename = "test_file_4.txt"
    with open(filename, "w") as f:
        f.write("!@#$%^&*()")
    input_dict = {"filename": filename, "name": "special_chars"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #os.remove(filename)

    # Input 5: File with newlines
    filename = "test_file_5.txt"
    with open(filename, "w") as f:
        f.write("Line 1\nLine 2\nLine 3")
    input_dict = {"filename": filename, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #os.remove(filename)
    
    # Input 6: File with spaces and tabs
    filename = "test_file_6.txt"
    with open(filename, "w") as f:
        f.write("  Space   and\tTab  ")
    input_dict = {"filename": filename, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #os.remove(filename)

    # Input 7: File with mixed content
    filename = "test_file_7.txt"
    with open(filename, "w") as f:
        f.write("Mixed content: 123 abc !@#")
    input_dict = {"filename": filename, "name": "mixed"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #os.remove(filename)

    # Input 8: Short filename
    filename = "short.txt"
    with open(filename, "w") as f:
        f.write("Short filename test")
    input_dict = {"filename": filename, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #os.remove(filename)

    # Input 9: Different name
    filename = "test_file_9.txt"
    with open(filename, "w") as f:
        f.write("Some text")
    input_dict = {"filename": filename, "name": "custom_name"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #os.remove(filename)

    # Input 10: File containing binary data.
    filename = "test_file_10.bin"
    binary_data = b'\x00\x01\x02\x03\x04\x05'
    with open(filename, "wb") as f:
        f.write(binary_data)
    input_dict = {"filename": filename, "name": "binary_file"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #os.remove(filename)
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.read_file"] = tf_io_read_file_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.read_file' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.read_file'.")

check_valid('tf.io.read_file', generated_inputs['tf.io.read_file'], lib="tf", suffix=0)
