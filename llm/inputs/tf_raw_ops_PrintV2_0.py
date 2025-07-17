
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_printv2_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array("Hello, world!", dtype=np.object_)
    output_stream = "stderr"
    end = "\n"
    name = "print_op_1"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array("This is a test.", dtype=np.object_)
    output_stream = "stdout"
    end = "\r\n"
    name = "print_op_2"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array("Another test.", dtype=np.object_)
    output_stream = "stderr"
    end = ""
    name = "print_op_3"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array("With a custom name.", dtype=np.object_)
    output_stream = "stdout"
    end = "!!!"
    name = "custom_name"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array("Using default output stream.", dtype=np.object_)
    output_stream = "stderr"
    end = "..."
    name = "print_op_5"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array("Empty end string.", dtype=np.object_)
    output_stream = "stderr"
    end = ""
    name = "print_op_6"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array("Testing different end characters.", dtype=np.object_)
    output_stream = "stderr"
    end = "\t"
    name = "print_op_7"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array("Long string for testing.", dtype=np.object_)
    output_stream = "stdout"
    end = "----"
    name = "print_op_8"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    input_tensor = np.array("String with special characters: !@#$%^&*()_+", dtype=np.object_)
    output_stream = "stderr"
    end = ".\n"
    name = "print_op_9"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array("A final test.", dtype=np.object_)
    output_stream = "stdout"
    end = "DONE"
    name = "print_op_10"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array("", dtype=np.object_)
    output_stream = "stdout"
    end = ""
    name = "print_op_11"
    input_dict = {"input": input_tensor, "output_stream": output_stream, "end": end, "name": name}
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
