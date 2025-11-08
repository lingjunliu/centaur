
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_WriteFile_inputs():
    list_of_inputs = []

    input_dict = {
        'name': 'string',
        'filename': np.array("test1.txt", dtype=np.str_),
        'contents': np.array("Hello, world!", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("/tmp/test2.txt", dtype=np.str_),
        'contents': np.array("Another test!", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test3.txt", dtype=np.str_),
        'contents': np.array("This is a longer string to test file writing.", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test4.txt", dtype=np.str_),
        'contents': np.array("1234567890", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test5.txt", dtype=np.str_),
        'contents': np.array("", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test6.txt", dtype=np.str_),
        'contents': np.array("Special characters: !@#$%^&*()_+=-`~[]\{}|;':\",./<>?", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test7.txt", dtype=np.str_),
        'contents': np.array("Unicode test: こんにちは世界", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test8.txt", dtype=np.str_),
        'contents': np.array("Newline test:\nThis is a new line.", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test9.txt", dtype=np.str_),
        'contents': np.array("Tab test:\tThis is a tab.", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test10.txt", dtype=np.str_),
        'contents': np.array("Number test: 123.456", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.WriteFile"] = tf_raw_ops_WriteFile_inputs()

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
