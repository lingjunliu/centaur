
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_raw_ops_WriteFile_inputs():
    list_of_inputs = []
    
    input_dict = {
        "filename": "/tmp/test1.txt",
        "contents": "Hello World",
        "name": "write_op1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test2.txt",
        "contents": "",
        "name": "write_op2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test3.txt",
        "contents": "Line 1\nLine 2\nLine 3",
        "name": "write_op3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test4.bin",
        "contents": "Binary content",
        "name": "write_op4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/nested/dir/test5.txt",
        "contents": "Nested directory content",
        "name": "write_op5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test6.txt",
        "contents": "Special chars: !@#$%^&*()",
        "name": "write_op6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test7.txt",
        "contents": "Unicode content",
        "name": "write_op7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test8.json",
        "contents": '{"key": "value", "number": 42}',
        "name": "write_op8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test9.txt",
        "contents": "A" * 1000,
        "name": "write_op9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test10.txt",
        "contents": "Tab\there\tand\nspaces   here",
        "name": "write_op10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test11.txt",
        "contents": "No operation name",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test12.csv",
        "contents": "col1,col2,col3\n1,2,3\n4,5,6",
        "name": "write_op12"
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
