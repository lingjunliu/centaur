
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import sys

def tf_print_inputs():
    list_of_inputs = []

    # Input 1: Basic example with numpy arrays
    inputs = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    output_stream = "sys.stdout"
    summarize = None
    sep = " "
    end = "\n"
    name = "print_basic"
    input_dict = {"inputs": inputs, "output_stream": "sys.stdout", "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different separator and end character
    inputs = [np.array([7, 8, 9]), np.array([10, 11, 12])]
    output_stream = "sys.stderr"
    summarize = 5
    sep = ", "
    end = "!"
    name = "print_sep_end"
    input_dict = {"inputs": inputs, "output_stream": "sys.stderr", "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Summarize with -1 to print all elements
    inputs = [np.array([[1, 2], [3, 4]])]
    output_stream = "sys.stdout"
    summarize = -1
    sep = " | "
    end = ""
    name = "print_summarize_all"
    input_dict = {"inputs": inputs, "output_stream": "sys.stdout", "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty list of inputs
    inputs = []
    output_stream = "sys.stderr"
    summarize = None
    sep = " - "
    end = "***"
    name = "print_empty_inputs"
    input_dict = {"inputs": inputs, "output_stream": "sys.stderr", "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: Multi-dimensional array
    inputs = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
    output_stream = "sys.stdout"
    summarize = 2
    sep = "  "
    end = "\n\n"
    name = "print_multi_dim"
    input_dict = {"inputs": inputs, "output_stream": "sys.stdout", "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List of different numpy array types
    inputs = [np.array([1, 2, 3], dtype=np.int32), np.array([1.1, 2.2, 3.3], dtype=np.float32), np.array([True, False, True], dtype=np.bool_)]
    output_stream = "sys.stderr"
    summarize = None
    sep = " | "
    end = "\n"
    name = "print_different_types"
    input_dict = {"inputs": inputs, "output_stream": "sys.stderr", "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values and zero
    inputs = [np.array([-1, 0, 1])]
    output_stream = "sys.stdout"
    summarize = None
    sep = " "
    end = "\n"
    name = "print_negative_values"
    input_dict = {"inputs": inputs, "output_stream": "sys.stdout", "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large array, check summarize
    inputs = [np.arange(100)]
    output_stream = "sys.stderr"
    summarize = 10
    sep = " "
    end = "\n"
    name = "print_large_array"
    input_dict = {"inputs": inputs, "output_stream": "sys.stderr", "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Check string name. tf.compat.v1.logging.info
    inputs = [np.array([1, 2, 3])]
    output_stream = "tf.compat.v1.logging.info"
    summarize = None
    sep = " "
    end = "\n"
    name = "testing_name_parameter"
    input_dict = {"inputs": inputs, "output_stream": "tf.compat.v1.logging.info", "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex multi-dimensional array and summarize
    inputs = [np.arange(24).reshape((2, 3, 4))]
    output_stream = "tf.compat.v1.logging.warning"
    summarize = 1
    sep = " "
    end = "\n"
    name = "print_complex_array"
    input_dict = {"inputs": inputs, "output_stream": "tf.compat.v1.logging.warning", "summarize": summarize, "sep": sep, "end": end, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: File output
    inputs = [np.array([9, 10, 11])]
    output_stream = "file:///tmp/test.txt"
    summarize = None
    sep = " "
    end = "\n"
    name = "print_file"
    input_dict = {"inputs": inputs, "output_stream": "file:///tmp/test.txt", "summarize": summarize, "sep": sep, "end": end, "name": name}
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
