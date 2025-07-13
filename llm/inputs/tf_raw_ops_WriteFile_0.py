
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_WriteFile_inputs():
    list_of_inputs = []

    # Input 1
    filename = np.array("test_file1.txt", dtype=np.string_)
    contents = np.array("Hello, world!", dtype=np.string_)
    name = None
    input_dict = {"filename": filename, "contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    filename = np.array("test_file2.txt", dtype=np.string_)
    contents = np.array("This is a longer string to write to the file.", dtype=np.string_)
    name = "WriteFileOp2"
    input_dict = {"filename": filename, "contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    filename = np.array("test_file3.txt", dtype=np.string_)
    contents = np.array("", dtype=np.string_)
    name = "WriteFileOp3"
    input_dict = {"filename": filename, "contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    filename = np.array("test_file4.txt", dtype=np.string_)
    contents = np.array("12345", dtype=np.string_)
    name = None
    input_dict = {"filename": filename, "contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    filename = np.array("path/to/test_file5.txt", dtype=np.string_)
    contents = np.array("This is in a subdirectory.", dtype=np.string_)
    name = "WriteFileOp5"
    input_dict = {"filename": filename, "contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    filename = np.array("./test_file6.txt", dtype=np.string_)
    contents = np.array("Relative path test.", dtype=np.string_)
    name = None
    input_dict = {"filename": filename, "contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Removed unicode to avoid encoding errors

    # Input 8
    filename = np.array("test_file8.txt", dtype=np.string_)
    contents = np.array("Special chars: !@#$%^&*()_+=-`~[]{}|;':\",./<>?", dtype=np.string_)
    name = None
    input_dict = {"filename": filename, "contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    filename = np.array("test_file9.txt", dtype=np.string_)
    contents = np.array("A very long string: " + "a" * 1000, dtype=np.string_)
    name = "WriteFileOp9"
    input_dict = {"filename": filename, "contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    filename = np.array("test_file10.txt", dtype=np.string_)
    contents = np.array("\nLine 1\nLine 2\nLine 3", dtype=np.string_)
    name = None
    input_dict = {"filename": filename, "contents": contents, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_WriteFile_inputs()
generated_inputs["tf.raw_ops.WriteFile"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.WriteFile"].append({
        "filename": input_dict["filename"],
        "contents": input_dict["contents"],
        "name": input_dict["name"]
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.WriteFile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.WriteFile'.")

check_valid('tf.raw_ops.WriteFile', generated_inputs['tf.raw_ops.WriteFile'], lib="tf", suffix=0)
