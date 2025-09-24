
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_write_file_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    input_dict = {
        "filename": np.array("test_file1.txt").astype(np.string_),
        "contents": np.array("Hello, world!").astype(np.string_),
        "name": "WriteFileOp1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty contents
    input_dict = {
        "filename": np.array("test_file2.txt").astype(np.string_),
        "contents": np.array("").astype(np.string_),
        "name": "WriteFileOp2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different filename
    input_dict = {
        "filename": np.array("test_file3.txt").astype(np.string_),
        "contents": np.array("Some different content").astype(np.string_),
        "name": "WriteFileOp3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Filename with path
    input_dict = {
        "filename": np.array("path/to/test_file4.txt").astype(np.string_),
        "contents": np.array("Content with path").astype(np.string_),
        "name": "WriteFileOp4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Long content string
    input_dict = {
        "filename": np.array("test_file5.txt").astype(np.string_),
        "contents": np.array("This is a very long string to test the WriteFile operation. It should be long enough to cover multiple blocks on disk.").astype(np.string_),
        "name": "WriteFileOp5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Special characters in filename
    input_dict = {
        "filename": np.array("test_file6!.txt").astype(np.string_),
        "contents": np.array("Special characters").astype(np.string_),
        "name": "WriteFileOp6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Special characters in content
    input_dict = {
        "filename": np.array("test_file7.txt").astype(np.string_),
        "contents": np.array("!@#$%^&*()_+=-`~[]{}|;':\",./<>?").astype(np.string_),
        "name": "WriteFileOp7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Unicode characters in content - Removed

    # Input 9: Number as string
    input_dict = {
        "filename": np.array("test_file9.txt").astype(np.string_),
        "contents": np.array("1234567890").astype(np.string_),
        "name": "WriteFileOp9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Filename with spaces
    input_dict = {
        "filename": np.array("test file10.txt").astype(np.string_),
        "contents": np.array("Filename with spaces").astype(np.string_),
        "name": "WriteFileOp10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.WriteFile"] = tf_raw_ops_write_file_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.WriteFile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.WriteFile'.")

check_valid('tf.raw_ops.WriteFile', generated_inputs['tf.raw_ops.WriteFile'], lib="tf", suffix=0)
