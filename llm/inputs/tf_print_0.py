
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import sys
import copy

def tf_print_inputs():
    list_of_inputs = []

    # Input 1
    inputs = [np.array([1, 2, 3, 4, 5])]
    output_stream = sys.stdout
    summarize = None
    sep = " "
    end = "\n"
    name = "print_tensor"
    input_dict = {"inputs": inputs, "output_stream": output_stream, "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
    output_stream = sys.stderr
    summarize = 2
    sep = ", "
    end = "!"
    name = None
    input_dict = {"inputs": inputs, "output_stream": output_stream, "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
    output_stream = "file:///tmp/test.txt"
    summarize = -1
    sep = ";"
    end = "END"
    name = "print_3d_tensor"
    input_dict = {"inputs": inputs, "output_stream": output_stream, "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [np.array([1, 2, 3])]
    output_stream = sys.stdout
    summarize = 1
    sep = "|"
    end = "||"
    name = "print_mixed"
    input_dict = {"inputs": inputs, "output_stream": output_stream, "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = [np.array([-1, -2, -3, -4, -5])]
    output_stream = sys.stderr
    summarize = 0
    sep = "---"
    end = "***"
    name = "print_negative"
    input_dict = {"inputs": inputs, "output_stream": output_stream, "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = [np.array([1])]
    output_stream = sys.stdout
    summarize = None
    sep = "..."
    end = ""
    name = "print_single"
    input_dict = {"inputs": inputs, "output_stream": output_stream, "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]
    output_stream = sys.stderr
    summarize = 1
    sep = "=="
    end = "++"
    name = "print_matrix"
    input_dict = {"inputs": inputs, "output_stream": output_stream, "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    output_stream = sys.stdout
    summarize = 2
    sep = " %% "
    end = " $$"
    name = "print_multiple"
    input_dict = {"inputs": inputs, "output_stream": output_stream, "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = [np.array([1.1, 2.2, 3.3, 4.4, 5.5])]
    output_stream = sys.stderr
    summarize = 3
    sep = "~~~"
    end = "^^^"
    name = "print_float"
    input_dict = {"inputs": inputs, "output_stream": output_stream, "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = [np.array([True, False, True])]
    output_stream = sys.stdout
    summarize = -1
    sep = ";;"
    end = "&&"
    name = "print_bool"
    input_dict = {"inputs": inputs, "output_stream": output_stream, "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.print"] = tf_print_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.print' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.print'.")

check_valid('tf.print', generated_inputs['tf.print'], lib="tf", suffix=0)
