
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_gfile_GFile_inputs():
    list_of_inputs = []

    # Input 1: Basic local file read
    input_dict = {"name": "/tmp/test_file.txt", "mode": "r"}
    with open("/tmp/test_file.txt", "w") as f:
        f.write("Hello, world!")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic local file write
    input_dict = {"name": "/tmp/test_file2.txt", "mode": "w"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Append mode
    input_dict = {"name": "/tmp/test_file3.txt", "mode": "a"}
    with open("/tmp/test_file3.txt", "w") as f:
        f.write("Initial content\n")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Read and write (r+)
    input_dict = {"name": "/tmp/test_file4.txt", "mode": "r+"}
    with open("/tmp/test_file4.txt", "w") as f:
        f.write("Some initial text.")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Write and read (w+)
    input_dict = {"name": "/tmp/test_file5.txt", "mode": "w+"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Append and read (a+)
    input_dict = {"name": "/tmp/test_file6.txt", "mode": "a+"}
    with open("/tmp/test_file6.txt", "w") as f:
        f.write("Existing text\n")
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Binary read mode (rb) - create file with binary content
    with open("/tmp/test_file7.bin", "wb") as f:
        f.write(b'\x00\x01\x02\x03')
    input_dict = {"name": "/tmp/test_file7.bin", "mode": "rb"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Binary write mode (wb)
    input_dict = {"name": "/tmp/test_file8.bin", "mode": "wb"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Binary read/write mode (r+b)
    with open("/tmp/test_file9.bin", "wb") as f:
        f.write(b'\x04\x05\x06\x07')
    input_dict = {"name": "/tmp/test_file9.bin", "mode": "r+b"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Binary append/read mode (a+b)
    with open("/tmp/test_file10.bin", "wb") as f:
        f.write(b'\x08\x09\x0a\x0b')
    input_dict = {"name": "/tmp/test_file10.bin", "mode": "a+b"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.gfile.GFile"] = tf_io_gfile_GFile_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.GFile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.GFile'.")

check_valid('tf.io.gfile.GFile', generated_inputs['tf.io.gfile.GFile'], lib="tf", suffix=0)
