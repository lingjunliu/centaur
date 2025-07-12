
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np
import os
import tempfile

def tf_io_read_file_inputs():
    list_of_inputs = []

    # Input 1: Valid filename
    temp = tempfile.NamedTemporaryFile(delete=False, dir="/tmp")
    filename = temp.name
    with open(filename, "w") as f:
        f.write("Test data 1")
    name = "read_file_op_1"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty file
    temp = tempfile.NamedTemporaryFile(delete=False, dir="/tmp")
    filename = temp.name
    open(filename, "w").close()
    name = "read_file_op_2"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Filename with spaces
    filename = "/tmp/test file 3.txt"
    with open(filename, "w") as f:
        f.write("Test data 3 with spaces")
    name = "read_file_op_3"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Long filename
    filename = "/tmp/" + "a" * 200 + ".txt"
    with open(filename, "w") as f:
        f.write("Long filename test")
    name = "read_file_op_4"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Filename with unicode characters
    filename = "/tmp/测试文件.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Unicode characters test")
    name = "read_file_op_5"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Filename with number
    filename = "/tmp/test42.txt"
    with open(filename, "w") as f:
        f.write("Test data with a number")
    name = "read_file_op_6"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Short name
    temp = tempfile.NamedTemporaryFile(delete=False, dir="/tmp")
    filename = temp.name
    with open(filename, "w") as f:
        f.write("Test data short")
    name = "s"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Long name
    temp = tempfile.NamedTemporaryFile(delete=False, dir="/tmp")
    filename = temp.name
    with open(filename, "w") as f:
        f.write("Test data long")
    name = "n" * 200
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Name with unicode characters
    temp = tempfile.NamedTemporaryFile(delete=False, dir="/tmp")
    filename = temp.name
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Test data with unicode name")
    name = "名称"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Name with numbers
    temp = tempfile.NamedTemporaryFile(delete=False, dir="/tmp")
    filename = temp.name
    with open(filename, "w") as f:
        f.write("Test data with name numbers")
    name = "name123"
    input_dict = {"filename": filename, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    for input_dict in list_of_inputs:
        if os.path.exists(input_dict["filename"]):
          os.remove(input_dict["filename"])

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
