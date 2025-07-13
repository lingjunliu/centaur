
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_printv2_inputs():
    list_of_inputs = []

    # Input 1: Basic string tensor
    input_tensor = np.array("Hello, TensorFlow!", dtype=np.bytes_)
    input_dict = {"input": input_tensor, "output_stream": "stderr", "end": "\n", "name": "print_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty string
    input_tensor = np.array("", dtype=np.bytes_)
    input_dict = {"input": input_tensor, "output_stream": "stdout", "end": "\r\n", "name": "print_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiline string
    input_tensor = np.array("This is a\nmultiline string.", dtype=np.bytes_)
    input_dict = {"input": input_tensor, "output_stream": "stderr", "end": "", "name": "print_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String with special characters
    input_tensor = np.array("String with \\ \" \t \n \r", dtype=np.bytes_)
    input_dict = {"input": input_tensor, "output_stream": "stdout", "end": "END", "name": "print_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different output_stream
    input_tensor = np.array("To log level INFO", dtype=np.bytes_)
    input_dict = {"input": input_tensor, "output_stream": "info", "end": "\n", "name": "print_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different output_stream
    input_tensor = np.array("To log level WARNING", dtype=np.bytes_)
    input_dict = {"input": input_tensor, "output_stream": "warning", "end": "\n", "name": "print_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different output_stream
    input_tensor = np.array("To log level ERROR", dtype=np.bytes_)
    input_dict = {"input": input_tensor, "output_stream": "error", "end": "\n", "name": "print_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String with numbers
    input_tensor = np.array("Numbers: 1234567890", dtype=np.bytes_)
    input_dict = {"input": input_tensor, "output_stream": "stderr", "end": "\n", "name": "print_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String with escape sequence
    input_tensor = np.array("String with tab: \t and newline: \n", dtype=np.bytes_)
    input_dict = {"input": input_tensor, "output_stream": "stderr", "end": "\n", "name": "print_op_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Long String
    long_string = "This is a very long string. " * 20
    input_tensor = np.array(long_string, dtype=np.bytes_)
    input_dict = {"input": input_tensor, "output_stream": "stderr", "end": "\n", "name": "print_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.PrintV2"] = tf_raw_ops_printv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.PrintV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PrintV2'.")

check_valid('tf.raw_ops.PrintV2', generated_inputs['tf.raw_ops.PrintV2'], lib="tf", suffix=0)
