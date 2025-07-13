
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_readfile_inputs():
    list_of_inputs = []

    names = [None, "my_read_op", None, None, None, None, "another_read_op", "very_very_very_long_op_name", "name with spaces", None]
    filenames = ["test_file_1.txt", "test_file_2.txt", "./path/to/test_file_3.txt", "", "test_file_with_!@#$%^&*.txt", "测试文件.txt", "another_test_file.txt", "long_file_name.txt", "file with spaces.txt", "./very/complex/path/to/a/very/complex/file.txt"]

    list_of_inputs = []

    for i in range(10):
        filename = tf.constant(filenames[i])
        name = names[i]

        input_dict = {
            "filename": filename,
            "name": name
        }
        list_of_inputs.append(copy.deepcopy(input_dict))
        
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ReadFile"] = tf_raw_ops_readfile_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ReadFile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReadFile'.")

check_valid('tf.raw_ops.ReadFile', generated_inputs['tf.raw_ops.ReadFile'], lib="tf", suffix=0)
